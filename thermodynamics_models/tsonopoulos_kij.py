"""Published binary interaction parameters for the Tsonopoulos correlation.

The characteristic binary constant is defined by

``Tc_ij = sqrt(Tc_i*Tc_j)*(1 - k_ij)``.

Specific fitted binaries take precedence over predictive family rules.  The
table is keyed by CAS number so process-local component symbols and aliases do
not affect the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TsonopoulosKijRecord:
    """A resolved Tsonopoulos binary parameter and its provenance."""

    value: float
    source: str
    method: str
    uncertainty: float | None = None
    temperature_range_K: tuple[float, float] | None = None


@dataclass(frozen=True)
class TsonopoulosPolarRecord:
    """Published pure-component polar parameters used to fit cross ``k_ij``."""

    a: float
    b: float
    source: str


_POLAR_PARAMETERS: dict[str, TsonopoulosPolarRecord] = {
    # Tsonopoulos & Dymond (1997), Table 3 and section 5.
    '7732-18-5': TsonopoulosPolarRecord(
        -0.0109, 0.0, 'Tsonopoulos and Heidman (1990), revised water fit'
    ),
    '67-56-1': TsonopoulosPolarRecord(
        0.0878, 0.0525, 'Tsonopoulos and Dymond (1997), Table 3'
    ),
    '64-17-5': TsonopoulosPolarRecord(
        0.0878, 0.0578, 'Tsonopoulos and Dymond (1997), Table 3'
    ),
    '71-23-8': TsonopoulosPolarRecord(
        0.0878, 0.0461, 'Tsonopoulos and Dymond (1997), Table 3'
    ),
    '71-36-3': TsonopoulosPolarRecord(
        0.0878, 0.0408, 'Tsonopoulos and Dymond (1997), Table 3'
    ),
    '111-27-3': TsonopoulosPolarRecord(
        0.0878, 0.0250, 'Tsonopoulos and Dymond (1997), Table 3'
    ),
    # Plyasunov & Shock (2003), Table 7.  These fitted a values are part of
    # the parameterization used to obtain that table's water-pair k_12.
    '7647-01-0': TsonopoulosPolarRecord(
        -0.011, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '7664-41-7': TsonopoulosPolarRecord(
        -0.022, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '74-87-3': TsonopoulosPolarRecord(
        -0.008, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '67-66-3': TsonopoulosPolarRecord(
        -0.003, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '75-00-3': TsonopoulosPolarRecord(
        -0.007, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '7446-09-5': TsonopoulosPolarRecord(
        -0.009, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
    '67-64-1': TsonopoulosPolarRecord(
        -0.032, 0.0, 'Plyasunov and Shock (2003), JCED 48, Table 7'
    ),
}


def published_tsonopoulos_polar_parameters(
    cas: str,
) -> TsonopoulosPolarRecord | None:
    """Return a fitted pure-component polar record when one is available."""
    return _POLAR_PARAMETERS.get(str(cas).strip())


def _pair(first: str, second: str) -> frozenset[str]:
    return frozenset((first, second))


_EXACT_BINARY_KIJS: dict[frozenset[str], TsonopoulosKijRecord] = {}


def _register(
    first: str,
    second: str,
    value: float,
    source: str,
    *,
    method: str = 'fitted binary',
    uncertainty: float | None = None,
    temperature_range_K: tuple[float, float] | None = None,
) -> None:
    """Register a value; later, newer sources deliberately supersede earlier ones."""
    _EXACT_BINARY_KIJS[_pair(first, second)] = TsonopoulosKijRecord(
        value=value,
        source=source,
        method=method,
        uncertainty=uncertainty,
        temperature_range_K=temperature_range_K,
    )


# Tsonopoulos (1974), AIChE Journal 20, Tables 3 and 4.  Strongly
# cross-associating pairs from Table 5 are deliberately excluded because the
# paper requires temperature-dependent k_ij values for them.
_SOURCE_1974 = 'Tsonopoulos (1974), AIChE Journal 20, Tables 3-4'
for _first, _second, _value in (
    ('67-56-1', '7727-37-9', 0.05),       # methanol / nitrogen
    ('67-56-1', '7440-37-1', 0.07),       # methanol / argon
    ('67-56-1', '74-82-8', 0.13),         # methanol / methane
    ('67-56-1', '74-85-1', 0.10),         # methanol / ethene
    ('67-56-1', '74-84-0', 0.12),         # methanol / ethane
    ('67-56-1', '124-38-9', 0.01),        # methanol / carbon dioxide
    ('67-56-1', '10024-97-2', 0.13),      # methanol / nitrous oxide
    ('67-64-1', '106-97-8', 0.07),        # acetone / n-butane
    ('67-64-1', '110-54-3', 0.13),        # acetone / n-hexane
    ('67-64-1', '110-82-7', 0.20),        # acetone / cyclohexane
    ('67-64-1', '75-15-0', 0.10),         # acetone / carbon disulfide
    ('67-64-1', '60-29-7', 0.10),         # acetone / diethyl ether
    ('78-93-3', '71-43-2', 0.12),         # 2-butanone / benzene
    ('96-22-0', '71-43-2', 0.12),         # 3-pentanone / benzene
    ('75-05-8', '110-82-7', 0.40),        # acetonitrile / cyclohexane
    ('60-29-7', '110-54-3', 0.08),        # diethyl ether / n-hexane
    ('60-29-7', '71-43-2', 0.10),         # diethyl ether / benzene
    ('67-56-1', '71-43-2', 0.20),         # methanol / benzene
    ('64-17-5', '71-43-2', 0.20),         # ethanol / benzene
    ('108-95-2', '7732-18-5', 0.15),      # phenol / water
):
    _register(_first, _second, _value, _SOURCE_1974)


# Tsonopoulos (1975), AIChE Journal 21, Table 2.  The chloroform /
# acetone and chloroform / ether complexes shown in Fig. 4 are excluded because
# their fitted k_ij is explicitly temperature dependent.
_SOURCE_1975 = 'Tsonopoulos (1975), AIChE Journal 21, Table 2'
for _first, _second, _value in (
    ('593-53-3', '7727-37-9', 0.08),      # methyl fluoride / nitrogen
    ('593-53-3', '124-38-9', 0.00),       # methyl fluoride / carbon dioxide
    ('74-87-3', '7440-37-1', 0.30),       # chloromethane / argon
    ('74-87-3', '75-15-0', 0.05),         # chloromethane / carbon disulfide
    ('74-87-3', '67-64-1', 0.00),         # chloromethane / acetone
    ('74-87-3', '74-83-9', 0.07),         # chloromethane / bromomethane
    ('75-00-3', '75-09-2', 0.07),         # chloroethane / dichloromethane
    ('75-00-3', '540-54-5', 0.02),        # chloroethane / 1-chloropropane
    ('75-00-3', '74-83-9', 0.07),         # chloroethane / bromomethane
    ('75-09-2', '67-64-1', -0.13),        # dichloromethane / acetone
    ('67-66-3', '110-54-3', 0.06),        # chloroform / n-hexane
    ('67-66-3', '71-43-2', 0.00),         # chloroform / benzene
    ('67-66-3', '56-23-5', -0.03),        # chloroform / carbon tetrachloride
    ('74-83-9', '7440-37-1', 0.20),       # bromomethane / argon
    ('74-83-9', '74-98-6', 0.05),         # bromomethane / propane
    ('74-83-9', '106-97-8', 0.05),        # bromomethane / n-butane
    ('74-88-4', '60-29-7', 0.05),         # iodomethane / diethyl ether
):
    _register(_first, _second, _value, _SOURCE_1975)


# Tsonopoulos (1978), AIChE Journal 24, Table 2 and text recommendations.
# The text recommends 0.25 for ammonia with N2, Ar, or Kr after considering
# the isolated argon result, so that recommendation is used here.
_SOURCE_1978 = 'Tsonopoulos (1978), AIChE Journal 24, Table 2 and text'
for _first, _second, _value in (
    ('7664-41-7', '1333-74-0', -0.45),    # ammonia / hydrogen
    ('7664-41-7', '7727-37-9', 0.25),     # ammonia / nitrogen
    ('7664-41-7', '7440-37-1', 0.25),     # ammonia / argon
    ('7664-41-7', '7439-90-9', 0.25),     # ammonia / krypton
    ('7664-41-7', '74-82-8', -0.09),      # ammonia / methane
    ('7664-41-7', '74-85-1', -0.13),      # ammonia / ethene
    ('7664-41-7', '74-86-2', -0.24),      # ammonia / acetylene
    ('7783-06-4', '74-84-0', 0.06),       # hydrogen sulfide / ethane
):
    _register(_first, _second, _value, _SOURCE_1978)
_register(
    '7783-06-4',
    '7732-18-5',
    0.15,
    _SOURCE_1978,
    method='phase-equilibrium estimate',
)


# Later specific additions and updates from Tsonopoulos (1979) and
# Tsonopoulos & Dymond (1997).
_SOURCE_1979 = 'Tsonopoulos (1979), Advances in Chemistry 182, Tables IV-V'
for _first, _second, _value in (
    ('64-17-5', '1333-74-0', 0.16),       # ethanol / hydrogen
    ('64-17-5', '7440-37-1', 0.17),       # ethanol / argon
    ('64-17-5', '74-82-8', 0.15),         # ethanol / methane
    ('64-17-5', '74-85-1', 0.15),         # ethanol / ethene
    ('64-17-5', '74-84-0', 0.17),         # ethanol / ethane
    ('64-17-5', '10024-97-2', 0.12),      # ethanol / nitrous oxide
    ('71-36-3', '7727-37-9', 0.24),       # 1-butanol / nitrogen
    ('71-36-3', '7440-37-1', 0.24),       # 1-butanol / argon
    ('71-36-3', '74-82-8', 0.20),         # 1-butanol / methane
    ('71-36-3', '74-84-0', 0.17),         # 1-butanol / ethane
    ('60-29-7', '7727-37-9', 0.22),       # diethyl ether / nitrogen
    ('60-29-7', '7440-37-1', 0.19),       # diethyl ether / argon
    ('60-29-7', '74-82-8', 0.12),         # diethyl ether / methane
    ('60-29-7', '74-84-0', 0.06),         # diethyl ether / ethane
    ('124-38-9', '85-01-8', 0.23),        # carbon dioxide / phenanthrene
    ('124-38-9', '120-12-7', 0.24),       # carbon dioxide / anthracene
    ('124-38-9', '56-23-5', 0.20),        # carbon dioxide / carbon tetrachloride
    ('124-38-9', '67-56-1', 0.01),        # carbon dioxide / methanol
    ('124-38-9', '64-17-5', 0.07),        # carbon dioxide / ethanol
    ('124-38-9', '71-36-3', 0.09),        # carbon dioxide / 1-butanol
    ('124-38-9', '60-29-7', -0.11),       # carbon dioxide / diethyl ether
):
    _register(_first, _second, _value, _SOURCE_1979)

_SOURCE_1997 = 'Tsonopoulos and Dymond (1997), Fluid Phase Equilibria 133'
_register('64-17-5', '110-54-3', 0.10, _SOURCE_1997)
_register('67-56-1', '60-29-7', 0.10, _SOURCE_1997)


# Plyasunov & Shock (2003), J. Chem. Eng. Data 48, Table 7.  This is
# the newest critical assessment in the source set and intentionally
# supersedes older water-pair values above.
_SOURCE_2003 = 'Plyasunov and Shock (2003), JCED 48, Table 7'
_water = '7732-18-5'
for _other, _value, _uncertainty, _range in (
    ('7440-37-1', 0.352, 0.027, (200.0, 1200.0)),
    ('7727-37-9', 0.296, 0.011, (243.0, 1000.0)),
    ('630-08-0', 0.251, 0.030, (311.0, 698.0)),
    # Table 7's upper limit is typographically truncated; the accompanying
    # data review states that the accepted N2O measurements span 298-373 K.
    ('10024-97-2', 0.17, 0.04, (298.0, 373.0)),
    ('7782-44-7', 0.40, 0.03, (298.0, 348.0)),
    ('74-82-8', 0.319, 0.014, (240.0, 698.0)),
    ('74-84-0', 0.360, 0.012, (298.0, 698.0)),
    ('74-98-6', 0.433, 0.013, (363.0, 698.0)),
    ('106-97-8', 0.443, 0.010, (363.0, 698.0)),
    ('109-66-0', 0.466, 0.013, (363.0, 698.0)),
    ('110-54-3', 0.494, 0.017, (363.0, 698.0)),
    ('142-82-5', 0.505, 0.015, (363.0, 698.0)),
    ('111-65-9', 0.517, 0.015, (363.0, 648.0)),
    ('124-38-9', 0.138, 0.015, (289.0, 1100.0)),
    ('74-85-1', 0.257, 0.017, (311.0, 648.0)),
    ('115-07-1', 0.32, 0.03, (363.0, 393.0)),
    ('71-43-2', 0.282, 0.011, (363.0, 698.0)),
    ('108-88-3', 0.27, 0.03, (400.0, 525.0)),
    ('110-82-7', 0.471, 0.013, (363.0, 698.0)),
    ('392-56-3', 0.342, 0.008, (378.0, 525.0)),
    ('7647-01-0', -0.224, 0.014, (263.0, 750.0)),
    ('67-56-1', 0.012, 0.008, (373.0, 527.0)),
    ('64-17-5', 0.05, 0.03, (400.0, 525.0)),
    ('7664-41-7', -0.171, 0.013, (373.0, 523.0)),
    ('74-87-3', 0.16, 0.03, (363.0, 423.0)),
    ('67-66-3', 0.20, 0.03, (353.0, 403.0)),
    ('75-00-3', 0.21, 0.03, (363.0, 423.0)),
    ('7446-09-5', 0.046, 0.030, (383.0, 483.0)),
    ('67-64-1', -0.020, 0.030, (383.0, 443.0)),
):
    _register(
        _water,
        _other,
        _value,
        _SOURCE_2003,
        uncertainty=_uncertainty,
        temperature_range_K=_range,
    )


_CLASS_DEFAULTS_1974 = {
    frozenset(('hydrocarbon', 'ketone')): 0.13,
    frozenset(('hydrocarbon', 'ether')): 0.10,
    frozenset(('hydrocarbon', 'alkanol')): 0.15,
    frozenset(('ketone', 'ether')): 0.13,
    frozenset(('ketone', 'alkanol')): 0.05,
    frozenset(('ketone', 'water')): 0.15,
}


def resolve_tsonopoulos_kij(
    cas_i: str,
    cas_j: str,
    family_i: str,
    family_j: str,
    critical_volume_i_cm3_mol: float,
    critical_volume_j_cm3_mol: float,
) -> TsonopoulosKijRecord | None:
    """Resolve a published fitted, correlated, or class-default ``k_ij``."""
    exact = _EXACT_BINARY_KIJS.get(_pair(cas_i, cas_j))
    if exact is not None:
        return exact

    families = frozenset((family_i, family_j))
    if family_i == family_j == 'alkane':
        numerator = 2.0 * (
            critical_volume_i_cm3_mol * critical_volume_j_cm3_mol
        ) ** (1.0 / 6.0)
        denominator = (
            critical_volume_i_cm3_mol ** (1.0 / 3.0)
            + critical_volume_j_cm3_mol ** (1.0 / 3.0)
        )
        value = 1.0 - (numerator / denominator) ** 3
        return TsonopoulosKijRecord(
            value=value,
            source='Tsonopoulos, Dymond, and Szafranski (1989), Eq. 13',
            method='critical-volume correlation',
        )

    if 'water' in families and ('alkane' in families or 'hydrocarbon' in families):
        hydrocarbon_volume = (
            critical_volume_j_cm3_mol
            if family_i == 'water'
            else critical_volume_i_cm3_mol
        )
        value = 0.6114 - 2.7135 / math.sqrt(hydrocarbon_volume)
        return TsonopoulosKijRecord(
            value=value,
            source='Tsonopoulos and Heidman (1990), Eq. 26',
            method='critical-volume correlation',
        )

    # Tsonopoulos & Dymond (1997) superseded the 1974 rough defaults for
    # alkanol/ether.  Its water/alkanol and water/ether recommendations remain
    # 0.10 and 0.35, respectively, when no specific fitted binary is known.
    revised_default = {
        frozenset(('alkanol', 'ether')): 0.10,
        frozenset(('alkanol', 'water')): 0.10,
        frozenset(('ether', 'water')): 0.35,
    }.get(families)
    if revised_default is not None:
        return TsonopoulosKijRecord(
            value=revised_default,
            source='Tsonopoulos and Dymond (1997), sections 7.5-7.7',
            method='class default',
        )

    default_families = frozenset(
        'hydrocarbon' if family == 'alkane' else family
        for family in (family_i, family_j)
    )
    if default_families == frozenset(('hydrocarbon', 'alkyl halide')):
        return TsonopoulosKijRecord(
            value=0.05,
            source='Tsonopoulos (1979), polar-nonpolar recommendation',
            method='class default',
        )

    class_default = _CLASS_DEFAULTS_1974.get(default_families)
    if class_default is not None:
        return TsonopoulosKijRecord(
            value=class_default,
            source='Tsonopoulos (1974), AIChE Journal 20, Table 6',
            method='class default',
        )
    return None


def exact_tsonopoulos_kij_records() -> dict[frozenset[str], TsonopoulosKijRecord]:
    """Return a copy of the published exact-binary table for auditing/tests."""
    return dict(_EXACT_BINARY_KIJS)
