"""Domalski--Hearing thermodynamic group additivity at 298.15 K.

This is a standalone implementation of E. S. Domalski and E. D. Hearing,
``J. Phys. Chem. Ref. Data`` **22** (1993) 805--1159.  It estimates standard
enthalpy of formation, heat capacity, and entropy in the gas, liquid, and
solid phases.  Entropy of formation, Gibbs energy of formation, and the
formation equilibrium constant are derived when the required primary values
are available.

The public entry point accepts only a SMILES string::

    >>> result = estimate("CCO")
    >>> result.gas.enthalpy_formation_kJ_mol
    -234.84

The implementation deliberately has no connection to pfdsim's property
resolver.  Missing cells in Table 2 remain ``None``; values are never copied
between phases or fitted locally.  Fragmentation is strict: every heavy atom
must be accounted for by a published group or structural correction.
"""

from __future__ import annotations

import math
import re
import itertools
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

TEMPERATURE_K = 298.15
PRESSURE_PA = 101_325.0

# Standard-state elemental entropies, J mol-1 K-1, from p. 808.  Values are
# per mole of elemental standard-state substance, hence the 1/2 factors for
# diatomic elements in ``_element_entropy_sum``.
ELEMENT_ENTROPIES = {
    6: 5.740,       # C(cr, graphite)
    1: 130.571,     # H2(g)
    8: 205.043,     # O2(g)
    7: 191.500,     # N2(g)
    16: 32.054,     # S(cr, rhombic)
    9: 202.682,     # F2(g)
    17: 222.972,    # Cl2(g)
    35: 152.21,     # Br2(l)
    53: 116.14,     # I2(cr)
}
_DIATOMIC_ELEMENTS = {1, 7, 8, 9, 17, 35, 53}
_ALLOWED_ATOMIC_NUMBERS = set(ELEMENT_ENTROPIES)


class DomalskiHearingError(ValueError):
    """Base error for the standalone estimator."""


class FragmentationError(DomalskiHearingError):
    """A structure cannot be represented by the published group scheme."""


@dataclass(frozen=True)
class Values:
    """A Table-2 row ordered as gas, liquid, and solid H/Cp/S values."""

    gas_h: float | None = None
    gas_cp: float | None = None
    gas_s: float | None = None
    liquid_h: float | None = None
    liquid_cp: float | None = None
    liquid_s: float | None = None
    solid_h: float | None = None
    solid_cp: float | None = None
    solid_s: float | None = None

    def phase(self, phase: str) -> tuple[float | None, float | None, float | None]:
        start = {"gas": 0, "liquid": 3, "solid": 6}[phase]
        row = (
            self.gas_h, self.gas_cp, self.gas_s,
            self.liquid_h, self.liquid_cp, self.liquid_s,
            self.solid_h, self.solid_cp, self.solid_s,
        )
        return row[start:start + 3]


def _v(*values: float | None) -> Values:
    if len(values) != 9:
        raise AssertionError("Domalski--Hearing rows must contain nine cells")
    return Values(*values)


# Table 2, pp. 816--826.  The notation is retained so fragmentation can be
# audited directly against the paper.  Blank published cells are None.
GROUP_VALUES: dict[str, Values] = {
    # CH groups
    "C-(H)4": _v(-74.48, 35.73, 206.92, None, None, None, None, None, None),
    "C-(H)3(C)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)2(C)2": _v(-20.63, 22.89, 39.16, -25.73, 30.42, 32.38, -29.41, 21.92, 23.01),
    "C-(H)(C)3": _v(-1.17, 20.08, -53.60, -4.77, 21.38, -23.89, -5.98, -48.81, -16.89),
    "C-(C)4": _v(19.20, 16.53, -149.49, 17.99, 10.24, -98.65, 12.47, -83.63, -33.19),
    "Cd-(H)2": _v(26.32, 21.38, 115.52, 21.75, 28.37, 86.19, 22.43, None, None),
    "Cd-(H)(C)": _v(36.32, 18.74, 33.05, 31.05, 24.60, 28.58, 25.48, None, None),
    "Cd-(C)2": _v(44.14, 15.10, -50.84, 39.16, 23.22, -29.83, 32.97, None, None),
    "Cd-(H)(Cd)": _v(28.28, 18.54, 27.74, 22.18, 31.67, 13.30, 17.53, 35.65, 21.75),
    "Cd-(C)(Cd)": _v(36.78, 17.57, -61.33, 30.42, 26.19, -41.92, 27.91, None, None),
    "Cd-(H)(CB)": _v(28.28, 18.54, 27.74, 22.18, 31.67, 13.30, 17.53, 35.65, 21.75),
    "Cd-(C)(CB)": _v(37.95, 15.90, -51.97, 38.58, None, None, None, None, None),
    "Cd-(CB)2": _v(32.88, None, None, 30.83, 25.10, None, 49.91, 32.50, None),
    "Cd-(Cd)(CB)": _v(None, None, None, None, None, None, 56.07, None, None),
    "C-(H)2(C)(Cd)": _v(-20.88, 20.63, 38.20, -25.73, 29.29, 31.67, -24.35, None, None),
    "C-(H)(C)2(Cd)": _v(-1.63, 27.49, -50.38, -5.02, 30.12, -28.07, -6.49, None, None),
    "C-(C)3(Cd)": _v(22.13, 9.16, -150.23, 20.79, 28.74, -108.20, 12.51, None, None),
    "C-(H)(C)(Cd)2": _v(-1.17, 20.08, -53.60, -4.77, 21.38, -23.89, -5.98, -48.81, -16.89),
    "C-(H)2(Cd)2": _v(-18.92, 24.77, 42.08, -24.43, 40.88, 19.32, -21.60, None, None),
    "C-(H)2(Cd)(CB)": _v(None, None, None, -24.73, None, None, None, None, None),
    "C-(H)(C)(Cd)(CB)": _v(None, None, None, -6.90, None, None, None, None, None),
    "Ct-(H)": _v(113.50, 22.55, 101.96, 104.47, 39.96, 67.57, 110.34, None, None),
    "Ct-(C)": _v(115.10, 13.22, 26.32, 107.15, 25.59, 14.25, 101.66, None, None),
    "Ct-(Cd)": _v(121.42, 10.71, 39.92, 114.77, None, None, None, None, None),
    "Ct-(CB)": _v(120.76, 10.17, 17.77, 119.00, None, None, 103.28, 32.30, None),
    "Ct-(Ct)": _v(120.76, 14.27, 25.94, 104.80, None, None, 103.28, None, None),
    "C-(H)2(C)(Ct)": _v(-19.70, 20.97, 42.80, -22.13, 30.39, 32.36, -29.41, None, None),
    "C-(H)(C)2(Ct)": _v(-3.16, 17.45, -45.69, None, None, None, None, None, None),
    "C-(C)3(Ct)": _v(None, None, None, 22.83, None, None, 26.38, None, None),
    "C-(H)2(Ct)2": _v(-41.14, None, None, -39.08, None, None, None, None, None),
    "C-(C)2(Ct)2": _v(None, None, None, 20.67, None, None, None, None, None),
    "Ca": _v(142.67, 15.86, 26.28, 134.68, 30.04, 14.39, 131.08, None, None),
    "CB-(H)(CB)2": _v(13.81, 13.61, 48.31, 8.16, 22.68, 28.87, 6.53, 20.13, 22.75),
    "CB-(C)(CB)2": _v(23.64, 9.75, -35.61, 19.16, 10.10, -19.50, 13.90, -23.26, -5.50),
    "CB-(Cd)(CB)2": _v(24.17, 14.12, -33.85, 19.12, 9.44, -9.04, 20.27, -20.00, -10.00),
    "CB-(Ct)(CB)2": _v(24.17, 14.12, -33.85, 19.12, 9.44, -9.04, 20.27, -20.00, -10.00),
    "CB-(CB)3": _v(21.66, 13.12, -36.57, 17.21, 17.07, None, 17.03, -1.72, -6.00),
    "C-(C)2(CB)2": _v(None, None, None, None, None, None, 52.81, None, None),
    "CBF-(CBF)(CB)2": _v(20.10, 0.00, 0.00, 15.83, 9.52, -5.54, 14.10, 2.30, -6.00),
    "CBF-(CB)(CBF)2": _v(16.00, None, None, 11.50, None, None, 12.00, 5.77, 2.00),
    "CBF-(CBF)3": _v(3.59, None, None, -0.90, None, None, 1.94, 8.00, 7.00),
    "CB-(CB)2(CBF)": _v(None, None, None, None, None, None, -8.77, None, None),
    "CB-(CB)(CBF)2": _v(22.46, None, None, None, None, None, 47.93, None, None),
    "CB-(O)(CB)2": _v(-4.75, 15.86, -43.72, -5.61, 39.71, -10.59, 1.00, -0.29, 1.59),
    "CB-(N)(CB)2": _v(-1.30, 16.07, -43.53, 1.50, 15.02, -24.43, 9.75, 13.00, -37.57),
    "C-(H)2(C)(CB)": _v(-21.34, 25.61, 42.59, -24.81, 22.90, 47.40, -22.10, 49.38, 26.90),
    "C-(H)(C)2(CB)": _v(-4.52, 22.45, -48.00, -5.82, 17.50, -13.90, -3.50, None, None),
    "C-(C)3(CB)": _v(18.28, 18.28, -147.19, 18.70, 5.17, -96.10, 21.57, None, None),
    "C-(H)2(CB)2": _v(-46.43, None, None, -26.50, 32.91, 51.97, -21.44, 69.06, 22.85),
    "C-(H)(C)(CB)2": _v(None, None, None, -21.47, 11.50, 28.12, 16.40, 43.55, None),
    "C-(H)(CB)3": _v(-6.86, None, None, None, None, None, 34.48, 63.64, -12.62),
    "C-(C)(CB)3": _v(None, None, None, None, None, None, 116.25, 39.83, None),
    "C-(CB)4": _v(27.04, None, None, None, None, None, 64.89, 58.74, None),

    # CHO groups
    "CO-(H)2": _v(-108.60, 35.40, 224.54, None, None, None, None, None, None),
    "CO-(C)(CO)": _v(-121.29, None, None, -135.04, None, None, -140.75, None, None),
    "CO-(H)(CO)": _v(-105.98, None, None, None, None, None, None, None, None),
    "CO-(CO)(CB)": _v(-112.30, None, None, None, None, None, -117.75, None, None),
    "CO-(O)(CO)": _v(-123.75, None, None, -123.30, 40.63, None, -120.81, None, None),
    "CO-(Cd)(O)": _v(-136.73, 24.56, 62.59, -155.56, 48.16, None, -134.10, 43.75, 32.90),
    "CO-(C)(O)": _v(-137.24, 24.56, 62.59, -149.37, 44.98, 32.72, -153.60, 44.98, 32.13),
    "CO-(H)(O)": _v(-124.39, 29.00, 147.03, -142.42, 65.10, 94.68, None, None, None),
    "CO-(O)2": _v(-111.88, None, None, -122.00, 31.46, None, -123.00, 4.25, -42.92),
    "CO-(H)(Cd)": _v(-126.96, None, None, -153.05, None, None, None, None, None),
    "CO-(H)(Ct)": _v(-126.96, None, None, -153.05, None, None, None, None, None),
    "CO-(CB)2": _v(-110.00, None, None, -119.00, None, None, -116.00, 109.33, None),
    "CO-(C)(CB)": _v(-148.82, None, None, -145.22, 73.35, None, -143.70, 71.38, 23.72),
    "CO-(H)(CB)": _v(-121.35, None, None, -138.12, 54.22, None, -160.18, None, None),
    "CO-(O)(CB)": _v(-125.00, None, None, -140.00, 48.16, None, -145.00, 43.75, 32.13),
    "CO-(C)2": _v(-132.67, 23.43, 64.31, -152.76, 52.97, 33.81, -157.95, None, None),
    "CO-(H)(C)": _v(-124.39, 29.00, 147.03, -142.42, 65.10, 93.55, None, None, None),
    "CO-(C)(Ct)": _v(None, None, None, None, 27.07, None, None, None, None),
    "O-(CO)2-aliphatic": _v(-214.50, -1.08, 34.16, -230.50, 5.28, None, -235.00, None, None),
    "O-(CO)2-aromatic": _v(-238.30, None, None, -220.90, None, None, -207.00, None, None),
    "O-(Cd)(CO)": _v(-198.03, None, None, -201.42, 19.58, None, None, None, None),
    "O-(C)(CO)": _v(-188.87, 11.80, 36.03, -196.02, 19.58, 38.28, -210.60, -6.00, 12.09),
    "O-(H)(CO)": _v(-254.30, 16.23, 101.71, -285.64, 37.82, 38.28, -282.15, 44.60, 21.78),
    "O-(CB)(CO)": _v(-167.00, None, None, -165.50, None, None, -170.00, 29.08, 45.32),
    "O-(C)(O)": _v(-20.75, None, None, -23.50, None, None, -30.20, None, None),
    "O-(H)(O)": _v(-72.26, None, None, -101.75, None, None, -105.30, None, None),
    "O-(Cd)2": _v(-139.29, None, None, -137.32, None, None, None, None, None),
    "O-(H)(Cd)": _v(None, None, None, None, 37.78, None, None, None, None),
    "O-(C)(Cd)": _v(-129.33, None, None, -133.72, 51.21, None, None, None, None),
    "O-(CB)2": _v(-77.66, None, None, -85.27, None, 23.31, -96.20, 15.90, 3.14),
    "O-(C)(CB)": _v(-92.55, None, None, -104.85, 8.10, None, -122.87, None, None),
    "O-(H)(CB)": _v(-160.30, 18.16, 121.50, -191.75, 44.64, 43.89, -199.25, 29.25, 28.62),
    "O-(C)2": _v(-101.42, 18.54, 29.33, -110.83, 24.27, 26.78, -119.00, None, None),
    "O-(H)(C)": _v(-159.33, 18.16, 121.50, -191.50, 44.64, 43.89, -199.66, 29.25, 28.62),
    "Cd-(O)(Cd)": _v(36.78, 17.57, -61.34, 30.42, 26.19, -41.92, 27.91, None, None),
    "Cd-(O)(C)": _v(44.14, 15.10, -50.84, 39.08, 23.22, -29.83, 32.97, None, None),
    "Cd-(O)(H)": _v(36.32, 18.74, 33.05, 31.05, 24.60, 28.58, 25.48, None, None),
    "Cd-(H)(CO)": _v(32.30, 15.61, 35.19, 26.61, 28.12, None, 7.82, -18.66, 27.53),
    "Cd-(C)(CO)": _v(None, None, None, None, 18.62, None, None, None, None),
    "Ct-(CO)": _v(None, None, None, None, None, None, 144.52, None, None),
    "CB-(CO)(CB)2": _v(15.50, None, None, 10.50, 4.39, None, 8.15, -42.89, 0.08),
    "C-(H)(CO)(C)(CB)": _v(None, None, None, None, None, None, 14.81, None, None),
    "C-(H)(CO)(CB)2": _v(None, None, None, None, None, None, 3.72, None, None),
    "C-(O)(CB)3": _v(None, None, None, None, None, None, 60.46, 57.49, None),
    "CO-(CO)(O)": _v(-88.00, None, None, -90.00, None, None, -80.50, None, None),
    "C-(C)2(O)(CB)": _v(15.30, None, None, 25.80, None, None, 29.30, None, None),
    "C-(H)(C)(O)2": _v(None, None, None, None, None, None, -52.50, None, None),
    "C-(H)2(C)(CO)": _v(-21.84, 24.69, 39.58, -24.14, 29.29, 39.87, -27.90, 21.92, 24.73),
    "C-(H)3(CO)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)(C)2(CO)": _v(-0.25, None, None, -3.89, 17.41, -24.52, -9.83, -80.51, None),
    "C-(CO)(C)3": _v(23.93, None, None, 26.15, 7.99, -85.98, 24.02, -114.10, None),
    "C-(H)2(CO)2": _v(-30.74, None, None, -23.06, 15.56, None, -19.10, None, None),
    "C-(H)2(CO)(Cd)": _v(-16.95, None, None, -19.62, None, None, None, None, None),
    "C-(H)2(CO)(Ct)": _v(-25.48, None, None, -26.61, None, None, None, None, None),
    "C-(H)2(CO)(CB)": _v(-16.20, None, None, -11.67, None, None, None, None, None),
    "C-(H)(O)(CO)(C)": _v(126.63, None, None, 123.43, 7.44, -46.71, -14.39, -58.45, 8.08),
    "C-(O)4": _v(-152.46, None, None, -133.34, None, None, None, None, None),
    "C-(H)(O)3": _v(-113.97, None, None, -107.74, 21.71, None, None, None, None),
    "C-(O)3(C)": _v(-114.39, None, None, -99.54, None, None, None, None, None),
    "C-(O)2(C)2": _v(-53.56, None, None, -41.30, None, None, None, None, None),
    "C-(H)(O)2(C)": _v(-57.78, None, None, -51.42, 12.38, None, None, None, None),
    "C-(H)2(O)2": _v(-62.22, None, None, -62.89, 39.92, 23.85, None, None, None),
    "C-(H)2(O)(CB)": _v(-33.76, None, None, -29.17, 46.48, None, None, None, None),
    "C-(H)2(O)(Cd)": _v(-27.49, 17.74, 37.49, -28.62, 41.30, None, None, None, None),
    "C-(H)2(C)(O)": _v(-32.90, 20.33, 43.43, -35.80, 33.64, 32.59, -33.00, 21.92, 24.73),
    "C-(H)(C)2(O)-ether": _v(-19.46, 17.78, -52.80, -21.00, 25.56, -25.31, -20.08, None, None),
    "C-(H)(C)2(O)-alcohol": _v(-26.10, 19.96, -43.05, -27.60, 49.83, -29.83, -29.08, 4.77, 6.95),
    "C-(C)3(O)-ether": _v(9.50, 14.60, -141.92, 0.79, 20.46, -94.68, -0.50, None, None),
    "C-(C)3(O)-alcohol": _v(-13.50, 15.73, -144.60, -11.13, 65.58, -122.48, -12.25, -85.48, -14.77),
    "C-(H)3(O)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),

    # CHN and CHNO groups
    "C-(H)3(N)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)2(C)(N)": _v(-28.30, 22.68, 42.26, -30.80, 30.42, 32.38, -34.00, 21.92, 23.01),
    "C-(H)(C)2(N)": _v(-16.70, 18.62, -63.55, -14.65, 28.28, -20.00, -13.90, None, None),
    "C-(C)3(N)": _v(0.29, 18.41, -152.59, 5.10, 19.66, -87.99, 1.00, -84.14, None),
    "C-(H)2(N)2": _v(-30.00, None, None, None, None, None, -26.00, None, None),
    "C-(H)2(CB)(N)": _v(-24.14, None, None, -26.09, 19.79, None, -33.31, None, None),
    "N-(H)2(C)": _v(19.25, 24.35, 124.40, 0.33, 62.59, 71.71, -6.30, 32.00, 39.00),
    "N-(H)2(C)-second-amino-acid": _v(19.25, 24.35, 126.90, 0.33, 62.59, 71.71, -46.00, 71.27, 48.75),
    "N-(H)(C)2": _v(67.55, 12.28, 33.96, 51.50, 59.37, 32.09, 47.80, -8.00, None),
    "N-(C)3": _v(116.50, 15.10, -61.71, 112.00, 26.11, -38.62, 101.00, -39.00, None),
    "N-(H)2(N)": _v(47.70, 26.36, 122.18, 25.30, 49.41, 60.58, 18.97, None, None),
    "N-(H)(C)(N)": _v(89.16, None, None, 75.00, 49.04, 22.05, None, None, None),
    "N-(C)2(N)": _v(120.71, None, None, 119.00, 41.67, -26.94, None, None, None),
    "N-(CB)2(N)": _v(None, None, None, None, None, None, 137.35, None, None),
    "N-(H)(CB)(N)": _v(87.50, None, None, 73.40, None, None, 66.90, None, None),
    "N-(CO)2(N)": _v(None, None, None, None, None, None, 73.62, None, None),
    "N-(H)(Cd)2": _v(83.55, None, None, 50.50, None, None, 45.40, None, None),
    "N-(C)(Cd)2": _v(120.64, None, None, 97.38, None, None, 88.92, None, None),
    "N-(H)(CB)2": _v(83.55, None, None, 50.50, None, None, 45.40, -3.00, None),
    "N-(H)2(CB)": _v(19.25, 24.35, 126.90, -11.00, 62.59, 71.71, -21.60, 26.00, 70.00),
    "N-(H)(C)(CB)": _v(59.00, None, None, 26.25, 65.20, None, 36.55, -50.00, None),
    "N-(C)2(CB)": _v(126.40, None, None, 109.40, 10.75, None, 96.50, -36.50, None),
    "N-(C)(CB)2": _v(120.44, None, None, 97.38, 7.95, None, 89.30, None, None),
    "Nr-(CB)": _v(69.00, 10.07, 47.01, 54.50, 19.75, 36.40, 57.00, None, None),
    "N-(CB)3": _v(123.15, None, None, 121.80, None, None, 107.50, -39.00, None),
    "Nr-(C)": _v(81.46, None, None, 73.68, None, None, None, None, None),
    "NA-(C)": _v(109.50, None, None, 104.85, None, None, 103.00, None, None),
    "NA-(CB)": _v(109.50, None, None, 104.85, None, None, 103.00, None, None),
    "NA-(oxide)(C)": _v(40.80, None, None, 22.65, None, None, None, None, None),
    "C-(H)2(C)(NA)": _v(-20.70, None, None, -25.70, None, None, -29.41, None, None),
    "C-(H)(C)2(NA)": _v(-2.66, None, None, -5.42, None, None, None, None, None),
    "C-(C)3(NA)": _v(11.50, None, None, 15.50, None, None, 10.50, None, None),
    "Cd-(H)(N)": _v(-16.00, None, None, -15.50, None, None, -13.00, None, None),
    "Cd-(C)(N)": _v(-5.74, None, None, -5.62, None, None, -3.95, None, None),
    "Cd-(H)(S)": _v(36.32, 18.74, 33.05, 31.05, 24.60, 28.58, 25.48, None, None),
    "Cd-(C)(S)": _v(45.73, 14.64, -51.92, None, None, None, None, None, None),
    "CB-(NO)(CB)2": _v(21.50, None, None, None, None, None, 23.00, None, None),
    "CB-(CN)(CB)2": _v(151.00, 41.09, 85.25, 122.38, 51.80, 64.75, 121.20, None, 50.45),
    "CB-(CNO)(CB)2": _v(-177.63, None, None, None, None, None, 155.69, None, None),
    "CB-(NA)(CB)2": _v(22.55, None, None, 20.08, None, None, 18.65, None, None),
    "Cd-(H)(N1)2": _v(6.30, None, None, None, None, None, 0.25, None, None),
    "CO-(H)(N)": _v(-124.39, 29.00, 147.03, -188.00, 65.10, 93.55, None, None, None),
    "CO-(C)(N)": _v(-133.26, 22.50, 56.70, -185.00, 49.16, None, -194.60, 39.00, 40.00),
    "CO-(CB)(N)": _v(None, None, None, None, None, None, -177.75, 111.50, None),
    "CO-(CB)(N)-amino-acid": _v(None, None, None, None, None, None, -177.75, 37.00, None),
    "CO-(Cd)(N)": _v(-171.80, None, None, None, None, None, None, None, None),
    "CO-(N)2": _v(-111.00, 32.40, 96.00, -190.50, None, None, -203.10, 124.00, 69.00),
    "N-(H)2(CO)": _v(-63.00, 17.00, 88.25, -63.90, 43.01, None, -65.25, -15.50, 18.00),
    "N-(H)2(CO)-amino-acid": _v(-63.00, None, None, -63.90, 43.01, None, -59.75, 45.88, 33.03),
    "N-(H)(C)(CO)": _v(-16.28, None, None, -17.10, 23.51, None, -9.80, -36.00, None),
    "N-(H)(C)(CO)-amino-acid": _v(-16.28, None, None, -17.10, 23.51, None, 5.50, 3.30, None),
    "N-(C)2(CO)": _v(45.00, None, None, 62.00, 13.93, None, 55.00, None, None),
    "N-(H)(CB)(CO)": _v(-20.84, None, None, None, None, None, -3.50, -41.00, None),
    "N-(H)(CO)2": _v(-91.00, None, None, None, None, None, -30.80, -157.02, None),
    "N-(C)(CO)2": _v(-11.64, None, None, 56.20, None, None, 64.00, None, None),
    "N-(CB)(CO)2": _v(9.12, None, None, None, None, None, None, None, None),
    "N-(CB)2(CO)": _v(None, None, None, None, None, None, 60.85, None, None),
    "N-(C)(CB)(CO)": _v(None, None, None, None, None, None, 72.00, None, None),
    "C-(H)3(CN)": _v(74.04, 52.22, 252.60, 40.56, 91.46, 149.62, None, None, None),
    "C-(H)2(C)(CN)": _v(94.52, 47.86, 167.25, 66.07, 83.01, 106.02, 69.85, 72.80, 96.15),
    "C-(H)(C)2(CN)": _v(113.50, 44.94, 67.86, 81.50, 83.09, None, 69.00, None, None),
    "C-(C)3(CN)": _v(137.96, None, None, 116.20, 69.91, -17.91, 102.07, None, None),
    "C-(C)2(CN)2": _v(None, None, None, None, None, None, None, 44.60, 74.57),
    "C-(H)2(Cd)(CN)": _v(95.31, None, None, 66.40, None, None, None, None, None),
    "Cd-(H)(CN)": _v(146.65, 42.38, 158.41, 117.28, 80.42, 92.72, None, None, None),
    "Ct-(CN)": _v(264.60, None, None, 250.20, None, None, None, None, None),
    "C-(H)3(NO2)": _v(-74.86, 57.32, 284.14, -112.60, 105.98, 171.75, None, None, None),
    "C-(H)2(NO2)2": _v(-58.90, None, None, -104.90, None, None, None, None, None),
    "C-(H)(NO2)3": _v(-0.30, None, None, -32.80, None, None, -48.00, None, None),
    "C-(NO2)4": _v(82.30, None, None, 38.30, None, None, None, None, None),
    "C-(H)2(C)(NO2)": _v(-60.50, 53.14, 203.60, -93.50, 97.74, None, -99.00, None, None),
    "C-(H)(C)2(NO2)": _v(-53.00, 49.58, 115.32, -82.50, None, None, -89.00, None, None),
    "C-(C)3(NO2)": _v(-36.65, None, None, -61.20, None, None, -76.55, None, None),
    "C-(H)2(CB)(NO2)": _v(-62.00, None, None, -82.76, None, None, -81.00, None, None),
    "C-(H)(C)(NO2)2": _v(-36.80, None, None, -88.80, None, None, -91.50, None, None),
    "C-(C)2(NO2)2": _v(-28.50, None, None, -77.20, None, None, -90.30, 71.38, None),
    "CB-(NO2)(CB)2": _v(-1.45, None, None, -28.30, 73.30, 79.95, -32.50, 50.96, 110.46),
    "O-(C)(NO)": _v(-24.23, 37.49, 166.11, -46.50, None, None, None, None, None),
    "O-(C)(NO2)": _v(-79.71, 51.46, 191.92, -108.96, 96.40, 127.50, -124.00, None, None),
    "N-(H)(C)(NO2)": _v(None, None, None, None, None, None, -16.50, 65.73, None),
    "N-(H)(CB)(NO2)": _v(None, None, None, None, None, None, None, -47.53, None),
    "N-(H)(CO)(NO2)": _v(None, None, None, None, None, None, -14.00, None, None),
    "N-(C)(NO2)2": _v(100.30, None, None, 53.50, None, None, None, None, None),
    "N-(C)(CB)(NO2)": _v(183.00, None, None, 167.00, None, None, 150.50, None, None),
    "N-(C)2(NO)": _v(90.00, None, None, 59.00, None, None, 55.00, None, None),
    "N-(C)2(NO2)": _v(88.00, None, None, 50.00, None, None, 40.00, None, None),
    "C-(H)2(C)(N3)": _v(None, None, None, 321.70, None, None, None, None, None),
    "C-(H)(C)2(N3)": _v(274.00, None, None, 255.00, None, None, None, None, None),
    "C-(H)2(CB)(N3)": _v(347.00, None, None, 327.40, None, None, None, None, None),
    "C-(CB)3(N3)": _v(328.60, None, None, None, None, None, 346.50, None, None),
    "CB-(N3)(CB)2": _v(320.00, None, None, 303.50, None, None, None, None, None),
    "C-(H)(C)(CO)(N)": _v(-18.70, None, None, None, None, None, -11.65, -22.85, -4.00),
    "C-(H)2(CO)(N)": _v(-3.10, None, None, None, None, None, -30.95, 21.92, 24.00),
    "C-(H)(CB)(CO)(N)": _v(None, None, None, None, None, None, None, 61.21, None),

    # CHS and CHSO groups
    "C-(H)3(S)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)2(C)(S)": _v(-23.17, 20.90, 41.87, -26.77, 24.18, 41.09, None, None, None),
    "C-(H)(C)2(S)": _v(-5.88, 20.29, -47.36, -6.07, 17.78, -16.61, None, None, None),
    "C-(C)3(S)": _v(13.52, 17.02, -145.38, 16.69, 8.88, -86.86, None, None, None),
    "C-(H)2(CB)(S)": _v(-18.53, None, None, -23.82, None, None, None, None, None),
    "C-(H)2(Cd)(S)": _v(-25.93, None, None, -32.44, None, None, None, None, None),
    "C-(H)2(S)2": _v(-25.10, None, None, None, None, None, None, None, None),
    "CB-(S)(CB)2": _v(-4.75, 15.86, 43.72, -5.61, 39.71, -10.59, 1.00, -0.29, 1.59),
    "S-(C)(H)": _v(18.64, 25.76, 137.67, 0.06, 51.34, 85.95, None, None, None),
    "S-(CB)(H)": _v(48.10, 20.98, 57.34, 28.51, 20.11, 89.04, None, None, None),
    "S-(C)2": _v(46.99, 22.64, 55.19, 29.82, 45.15, 29.80, None, None, None),
    "S-(C)(CB)": _v(76.21, None, None, 58.20, 16.43, 35.44, 42.00, None, None),
    "S-(H)(Cd)": _v(25.52, None, None, None, None, None, None, None, None),
    "S-(C)(Cd)": _v(54.39, None, None, None, None, None, None, None, None),
    "S-(Cd)2": _v(102.60, 20.04, 68.59, None, None, None, None, None, None),
    "S-(CO)(C)": _v(76.21, None, None, 58.20, 16.43, 35.44, 42.00, None, None),
    "S-(CB)2": _v(102.60, 20.04, 68.59, 93.02, -35.10, None, None, None, None),
    "S-(C)(S)": _v(27.62, 23.25, 50.50, 14.36, 40.71, 30.84, None, None, None),
    "S-(CB)(S)": _v(57.45, None, None, None, None, None, None, 40.60, None),
    "S-(S)2": _v(12.59, 19.66, 56.07, None, None, None, None, None, None),
    "S-(H)(S)": _v(7.95, None, None, None, None, None, None, None, None),
    "S-(H)(CO)": _v(-5.90, 31.92, 130.54, None, None, None, None, None, None),
    "CO-(C)(S)": _v(-132.67, 23.43, 64.31, -152.76, 52.97, 33.81, None, None, 33.89),
    "SO-(C)2": _v(-66.78, 37.15, 75.73, -108.98, 80.22, 22.18, None, None, None),
    "SO-(CB)2": _v(-62.26, None, None, None, None, None, None, None, None),
    "SO-(C)(CB)": _v(-72.00, None, None, None, None, None, None, None, None),
    "C-(H)3(SO)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)2(C)(SO)": _v(-29.16, None, None, -36.88, None, None, None, None, None),
    "C-(H)(C)2(SO)": _v(None, None, None, None, None, None, None, None, None),
    "C-(C)3(SO)": _v(4.56, None, None, 0.97, None, None, None, None, None),
    "C-(H)2(Cd)(SO)": _v(-27.56, None, None, -32.63, None, None, None, None, None),
    "CB-(SO)(CB)2": _v(15.48, None, None, 25.44, 4.39, None, 7.55, -42.89, 0.08),
    "SO-(O)2": _v(-213.00, None, None, None, None, None, None, None, None),
    "O-(SO)(H)": _v(-158.60, None, None, None, None, None, None, None, None),
    "O-(C)(SO)": _v(-92.60, None, None, None, None, None, None, None, None),
    "SO2-(C)2": _v(-288.58, 48.54, 87.37, -341.14, None, None, -356.62, -9.55, 32.10),
    "SO2-(CB)2": _v(-287.76, None, None, None, None, None, -305.40, None, None),
    "SO2-(C)(CB)": _v(-289.10, None, None, None, None, None, None, None, None),
    "SO2-(Cd)(CB)": _v(-291.55, None, None, None, None, None, None, None, None),
    "SO2-(Cd)2": _v(-306.70, None, None, None, None, None, None, None, None),
    "SO2-(O)2": _v(-417.30, None, None, None, None, None, None, None, None),
    "SO2-(SO2)(CB)": _v(-325.18, None, None, None, None, None, -361.75, None, None),
    "SO2-(C)(Cd)": _v(-316.80, None, None, None, None, None, None, None, None),
    "SO2-(Ct)(CB)": _v(-296.30, None, None, None, None, None, None, None, None),
    "O-(SO2)(H)": _v(-158.60, None, None, None, None, None, None, None, None),
    "O-(C)(SO2)": _v(-91.40, None, None, None, None, None, None, None, None),
    "C-(H)3(SO2)": _v(-42.26, 25.73, 127.32, -47.61, 36.48, 83.30, -46.74, 67.45, 56.69),
    "C-(H)2(C)(SO2)": _v(-27.03, None, None, -33.76, None, None, -35.96, None, None),
    "C-(H)(C)2(SO2)": _v(-14.00, None, None, None, None, None, None, None, None),
    "C-(C)3(SO2)": _v(1.52, None, None, 2.00, None, None, 3.78, None, None),
    "C-(H)2(Cd)(SO2)": _v(-29.49, None, None, -49.05, None, None, None, None, None),
    "C-(H)(C)(Cd)(SO2)": _v(-71.99, None, None, None, None, None, None, None, None),
    "C-(H)2(CB)(SO2)": _v(-29.80, None, None, None, None, None, None, None, None),
    "C-(H)2(Ct)(SO2)": _v(16.36, None, None, None, None, None, None, None, None),
    "CB-(SO2)(CB)2": _v(15.48, None, None, 25.44, 4.39, None, 7.55, -42.89, 0.08),
    "Cd-(H)(SO2)": _v(51.58, None, None, None, None, None, None, None, None),
    "Cd-(C)(SO2)": _v(64.01, None, None, None, None, None, None, None, None),
    "Ct-(SO2)": _v(177.10, None, None, None, None, None, None, None, None),

    # CHX and CHXO groups.  A halogen is included in its carbon group.
    "C-(H)3(F)": _v(-247.00, 37.49, 231.93, None, None, None, None, None, None),
    "C-(H)3(Cl)": _v(-81.90, 40.75, 243.60, None, None, None, None, None, None),
    "C-(H)3(Br)": _v(-37.66, 42.43, 254.94, -61.10, None, None, None, None, None),
    "C-(H)3(I)": _v(14.30, 44.14, 263.14, -11.70, 82.76, None, None, None, None),
    "C-(C)(F)3": _v(-673.81, 52.99, 178.22, -709.07, 73.18, 135.56, None, None, None),
    "C-(H)2(C)(F)": _v(-221.12, 33.66, 146.80, None, None, None, None, None, None),
    "C-(H)(C)2(F)": _v(-204.46, 30.55, 55.76, None, None, None, None, None, None),
    "C-(C)3(F)": _v(-202.92, None, None, None, None, None, None, None, None),
    "C-(H)(C)(F)2": _v(-454.74, 42.22, 164.32, -487.23, 68.04, None, None, None, None),
    "C-(C)2(F)2": _v(-411.39, 41.42, 74.48, -400.37, None, None, -428.77, None, None),
    "C-(C)(Cl)(F)2": _v(-462.70, 57.32, 169.45, -466.00, 83.64, 138.31, None, None, None),
    "C-(H)(C)(Cl)(F)": _v(-271.14, None, None, None, None, None, None, None, None),
    "C-(C)(Cl)3": _v(-81.98, 68.18, 202.14, -112.93, 102.20, 145.91, None, None, None),
    "C-(H)(C)(Cl)2": _v(-79.10, 50.69, 183.28, -102.60, 85.02, 128.45, None, None, None),
    "C-(H)2(C)(Cl)": _v(-69.45, 37.53, 159.24, -86.90, 63.76, 104.27, -85.65, None, None),
    "C-(H)(C)2(Cl)": _v(-55.61, 35.00, 71.34, -71.17, 66.02, None, None, None, None),
    "C-(C)3(Cl)": _v(-43.70, 29.63, -24.26, -56.78, None, None, None, None, None),
    "C-(C)2(Cl)2": _v(-79.56, 54.40, 95.41, -101.80, 74.24, None, None, None, None),
    "C-(C)(Br)3": _v(None, 69.87, 233.05, None, None, None, None, None, None),
    "C-(H)(C)(Br)2": _v(None, None, None, None, None, None, None, None, None),
    "C-(H)2(C)(Br)": _v(-21.78, 37.82, 173.31, -42.65, 66.00, 113.00, None, None, None),
    "C-(H)(C)2(Br)": _v(-10.75, 36.77, 84.69, -27.31, 59.24, None, None, None, None),
    "C-(C)3(Br)": _v(7.26, 39.33, -13.46, -7.40, None, None, None, None, None),
    "C-(C)2(Br)2": _v(None, None, None, None, None, None, None, None, None),
    "C-(C)(I)3": _v(None, None, None, None, None, None, None, None, None),
    "C-(H)(C)(I)2": _v(108.78, 51.04, 228.45, None, None, None, None, None, None),
    "C-(H)2(C)(I)": _v(33.54, 40.94, 177.78, 4.14, 65.36, None, 3.65, None, None),
    "C-(H)(C)2(I)": _v(48.74, 38.62, 88.10, 24.78, None, None, None, None, None),
    "C-(C)3(I)": _v(68.46, 41.09, -3.21, 48.60, None, None, None, None, None),
    "C-(C)2(I)2": _v(None, None, None, None, None, None, None, None, None),
    "C-(H)(C)(Br)(Cl)": _v(-18.45, 51.88, 191.21, None, None, None, None, None, None),
    "C-(H)(C)(Cl)(O)": _v(-90.37, 37.66, 66.53, None, None, None, None, None, None),
    "C-(H)2(I)(O)": _v(15.90, None, 170.29, None, None, None, None, None, None),
    "N-(C)(F)2": _v(-32.64, None, None, None, None, None, None, None, None),
    "C-(C)(Cl)2(F)": _v(-322.54, None, None, -343.87, 89.29, 141.71, None, None, None),
    "C-(C)(Br)(F)2": _v(-394.55, None, None, None, 85.40, 149.70, None, None, None),
    "C-(C)(Br)2(F)": _v(None, None, None, None, None, None, None, None, None),
    "C-(Br)(Cl)(F)": _v(None, None, None, None, None, None, None, None, None),
    "Cd-(H)(F)": _v(-165.12, 28.45, 137.24, None, None, None, None, None, None),
    "Cd-(H)(Cl)": _v(4.37, 32.75, 147.85, -12.67, 56.62, None, None, None, None),
    "Cd-(H)(Br)": _v(50.94, 34.10, 159.91, None, 79.13, None, None, None, None),
    "Cd-(H)(I)": _v(102.36, 36.82, 169.45, None, None, None, None, None, None),
    "Cd-(C)(Cl)": _v(-5.06, None, 62.76, -2.23, None, None, None, None, None),
    "Cd-(F)2": _v(-329.90, 39.43, 155.63, None, None, None, None, None, None),
    "Cd-(Cl)2": _v(-11.51, 46.86, 175.41, -32.08, 76.47, 115.35, None, None, None),
    "Cd-(Br)2": _v(None, 51.46, 199.16, None, None, None, None, None, None),
    "Cd-(Cl)(F)": _v(-235.10, 44.50, 175.61, None, None, None, None, None, None),
    "Cd-(Br)(F)": _v(None, 45.19, 177.82, None, None, None, None, None, None),
    "Cd-(Cl)(Br)": _v(None, 50.63, 188.70, None, None, None, None, None, None),
    "Cd-(I)2": _v(None, None, None, None, None, None, None, None, None),
    "Ct-(F)": _v(None, None, None, None, None, None, None, None, None),
    "Ct-(Cl)": _v(None, 33.01, 140.00, None, None, None, None, None, None),
    "Ct-(Br)": _v(None, 34.69, 151.30, None, None, None, None, None, None),
    "Ct-(I)": _v(None, 35.53, 158.41, None, None, None, None, None, None),
    "CB-(F)(CB)2": _v(-181.26, 26.10, 67.52, -191.20, 37.09, 54.19, -194.00, 32.05, 39.79),
    "CB-(Cl)(CB)2": _v(-17.03, 29.33, 77.08, -32.20, 35.27, 55.47, -32.00, 33.55, 43.37),
    "CB-(Br)(CB)2": _v(36.35, 29.65, 88.60, 19.90, 40.91, 74.85, 13.50, None, 54.45),
    "CB-(I)(CB)2": _v(94.50, 32.70, 98.26, 73.70, 45.17, 61.08, 70.40, 40.08, None),
    "C-(H)2(CO)(Cl)": _v(-44.26, None, None, -58.41, None, None, -74.75, None, None),
    "C-(H)(CO)(Cl)2": _v(-40.40, None, None, -55.11, None, None, None, None, None),
    "CO-(C)(F)": _v(-379.84, None, None, -419.59, None, None, None, None, None),
    "C-(CB)(F)3": _v(-691.79, 52.30, 179.08, -696.66, None, None, None, None, None),
    "C-(H)2(CB)(Br)": _v(-29.49, None, None, -44.06, None, None, None, None, None),
    "C-(H)2(CO)(Br)": _v(-29.49, None, None, -44.06, None, None, None, None, None),
    "C-(H)2(CB)(I)": _v(7.31, None, None, -7.24, None, None, None, None, None),
    "C-(H)2(CB)(Cl)": _v(-73.79, None, None, -92.56, None, None, None, None, None),
    "CO-(C)(Cl)": _v(-200.54, 42.09, 176.66, -225.29, 80.67, None, None, None, None),
    "CO-(CB)(Cl)": _v(None, None, None, -216.67, 69.21, None, -212.99, None, None),
    "CO-(C)(Br)": _v(-148.54, None, None, -175.49, None, None, None, None, None),
    "CO-(C)(I)": _v(-83.94, None, None, -117.09, None, None, None, None, None),
    "C-(H)(C)(CO)(Cl)": _v(-39.88, None, None, -35.46, 49.45, None, None, None, None),
    "C-(C)(CO)(Cl)2": _v(None, None, None, None, 74.22, None, None, None, None),
}


CORRECTION_VALUES: dict[str, Values] = {
    "CH3-tertiary": _v(-2.26, 0.0, 0.0, -2.18, 0.0, 0.0, -2.34, 0.0, 0.0),
    "CH3-quaternary": _v(-4.56, 0.0, 0.0, -4.39, 0.0, 0.0, -4.35, 0.0, 0.0),
    "CH3-tert-quat": _v(-1.80, 0.0, 0.0, -1.77, 0.0, 0.0, -2.70, 0.0, 0.0),
    "CH3-quat-quat": _v(-0.64, 0.0, 0.0, -0.64, 0.0, 0.0, -2.24, 0.0, 0.0),
    "cis-unsaturation": _v(4.85, -8.03, 5.06, 5.27, 0.0, 0.0, 5.73, 0.0, 0.0),
    "tert-butyl-cis": _v(17.24, 0.0, 0.0, 17.48, 0.0, 0.0, 17.57, 0.0, 0.0),
    "ortho-hydrocarbon": _v(1.26, 6.40, -2.50, 3.26, 3.50, 0.0, 5.0, 0.0, 0.0),
    "meta-hydrocarbon": _v(-0.63, 0.71, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    "cyclopropane-unsub-rsc": _v(115.15, -12.73, 134.86, 111.58, -28.53, None, None, None, None),
    "cyclopropane-sub-rsc": _v(105.95, None, None, 96.58, None, None, None, None, None),
    "cyclobutane-rsc": _v(110.89, -19.34, 126.04, 106.64, -10.68, 51.48, 114.43, None, None),
    "cyclopentane-unsub-rsc": _v(26.75, -31.44, 116.22, 22.84, -23.32, 42.24, 34.0, None, None),
    "cyclopentane-sub-rsc": _v(19.55, -27.87, 118.39, 23.59, -23.32, 56.65, 34.0, None, None),
    "cyclohexane-unsub-rsc": _v(0.68, -31.07, 78.18, -1.77, -26.21, 10.07, 10.94, None, None),
    "cyclohexane-sub-rsc": _v(-0.39, -22.82, 83.97, -2.06, -26.21, 25.10, 10.30, None, None),
    "cycloheptane-rsc": _v(26.34, -37.14, 73.97, 23.50, -32.19, 15.89, None, None, None),
    "cyclooctane-rsc": _v(40.65, -43.17, 70.78, 38.10, -27.88, 2.96, None, None, None),
    "cyclononane-rsc": _v(52.91, None, None, 50.40, None, None, None, None, None),
    "cyclodecane-rsc": _v(51.99, None, None, 50.61, None, None, None, None, None),
    "cycloundecane-rsc": _v(47.56, None, None, 47.55, None, None, None, None, None),
    "cyclododecane-rsc": _v(17.31, None, None, None, None, None, 46.27, None, None),
    "cyclotridecane-rsc": _v(21.84, None, None, 24.83, None, None, None, None, None),
    "cyclotetradecane-rsc": _v(49.37, None, None, None, None, None, 37.48, None, None),
    "cyclopentadecane-rsc": _v(8.03, None, None, None, None, None, 65.09, None, None),
    "cyclohexadecane-rsc": _v(8.41, None, None, None, None, None, 67.14, None, None),
    "cycloheptadecane-rsc": _v(-13.59, None, None, None, None, None, 69.56, None, None),
    "cyclopropene-rsc": _v(223.26, None, None, None, None, None, None, None, None),
    "cyclobutene-rsc": _v(125.81, -11.67, 126.77, None, None, None, None, None, None),
    "cyclopentene-unsub-rsc": _v(24.18, -26.53, 113.76, 21.45, -15.82, 48.37, None, None, None),
    "cyclopentene-sub-rsc": _v(24.31, -24.50, 117.11, 19.82, -15.82, 48.37, None, None, None),
    "cyclohexene-rsc": _v(5.61, -19.50, 95.69, 2.04, -20.26, 29.34, None, None, None),
    "cycloheptene-rsc": _v(21.81, None, None, None, None, None, None, None, None),
    "cyclooctene-rsc": _v(24.65, None, None, 18.26, None, None, None, None, None),
    "1,3-cyclopentadiene-rsc": _v(24.07, None, None, 23.95, None, None, None, None, None),
    "1,3-cyclohexadiene-rsc": _v(17.14, None, None, 16.41, -26.56, 50.18, None, None, None),
    "1,4-cyclohexadiene-rsc": _v(-2.69, None, None, -5.64, -34.22, 36.41, None, None, None),
    "1,3-cycloheptadiene-rsc": _v(27.54, None, None, None, None, None, None, None, None),
    "1,5-cyclooctadiene-rsc": _v(39.34, None, None, 36.42, -7.45, 23.35, None, None, None),
    "1,3,5-cycloheptatriene-rsc": _v(16.84, -18.63, 102.26, 18.59, -54.00, 84.96, None, None, None),
    "cyclooctatetraene-rsc": _v(71.37, -26.31, 116.38, 77.07, -68.18, 113.89, None, None, None),
    "spiropentane-rsc": _v(248.50, -19.97, 286.59, 242.58, 2.60, 162.81, None, None, None),
    "naphthalene-unsub": _v(0.0, 11.83, -19.66, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "naphthalene-1-sub": _v(0.0, 14.39, -21.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "naphthalene-2-sub": _v(0.0, 16.48, -23.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "cis-decalin-rsc": _v(127.00, -183.44, None, -4.02, -54.12, 53.75, None, None, None),
    "trans-decalin-rsc": _v(123.81, -183.24, None, -15.22, -57.63, 53.67, None, None, None),
    "cis-hexahydroindan-rsc": _v(19.51, None, None, 16.39, -41.52, 86.59, None, None, None),
    "trans-hexahydroindan-rsc": _v(15.16, None, None, 13.29, -46.00, 79.98, None, None, None),
    "2,2-metacyclophane-rsc": _v(52.08, None, None, None, None, None, 55.06, -24.92, None),
    "2,2-metaparacyclophane-rsc": _v(99.35, None, None, None, None, None, 109.46, -4.02, None),
    "2,2-paracyclophane-rsc": _v(125.09, None, None, None, None, None, 127.26, -13.18, -1.92),
    "3,3-paracyclophane-rsc": _v(50.95, None, None, None, None, None, 65.53, 14.90, None),
    "adamantane-rsc": _v(-6.14, None, None, None, None, None, 3.18, None, None),
    "fluoranthene-rsc": _v(63.21, 62.47, 59.66, -4.13, -5.92, None, None, None, None),
    "bicyclo[2.2.2]octane-rsc": _v(27.12, None, None, None, -67.59, -63.45, 41.52, None, None),
    "bicyclo[3.3.3]undecane-rsc": _v(99.06, None, None, None, None, None, 124.10, None, None),
    "cis-bicyclo[6.1.0]nonane-rsc": _v(115.55, None, None, 109.35, None, None, None, None, None),
    "bicyclo[1.1.0]butane-rsc": _v(260.70, None, None, 254.70, None, None, None, None, None),
    "bicyclo[3.1.0]hexane-rsc": _v(123.16, None, None, 117.56, None, None, None, None, None),
    "bicyclo[2.2.1]hepta-2,5-diene-rsc": _v(125.29, None, None, 124.87, None, None, None, None, None),
    "tetracyclo[3.2.0.2,7.0.4,6]heptane-rsc": _v(366.75, None, None, 356.45, None, None, None, None, None),
    "tricyclo[2.2.1.0.2,6]heptane-rsc": _v(148.67, None, None, 139.67, None, None, None, None, None),
    "bicyclo[2.2.1]hept-2-ene-rsc": _v(82.79, None, None, 73.58, None, None, 102.73, None, None),
    "bicyclo[2.2.1]heptane-rsc": _v(43.49, None, None, 45.39, None, None, 57.01, None, None),
    "bicyclo[4.1.0]heptane-rsc": _v(106.99, None, None, 101.39, None, None, None, None, None),
    "pentacyclo[4.2.0.2,5.0.3,8.0.4,7]octane-rsc": _v(674.60, None, None, None, None, None, 632.84, None, None),
    "bicyclo[2.2.2]oct-2-ene-rsc": _v(33.64, None, None, None, None, None, 56.36, None, None),
    "bicyclo[4.2.0]octane-rsc": _v(100.72, None, None, 95.72, None, None, None, None, None),
    "bicyclo[5.1.0]octane-rsc": _v(109.42, None, None, 103.62, None, None, None, None, None),
    "trans-bicyclo[6.1.0]nonane-rsc": _v(107.05, None, None, 107.25, None, None, None, None, None),
    "bicyclo[3.3.1]nonane-rsc": _v(19.25, None, None, None, None, None, 39.63, None, None),
    "cis-bicyclo[3.3.0]octane-rsc": _v(33.22, None, None, 27.92, None, None, None, None, None),
    "trans-bicyclo[3.3.0]octane-rsc": _v(59.52, None, None, 54.72, None, None, None, None, None),
    "cyclopentanone-rsc": _v(22.85, None, None, 15.10, None, None, None, None, None),
    "cyclohexanone-rsc": _v(10.50, -31.82, 66.98, 5.60, -25.61, 11.29, None, None, None),
    "cycloheptanone-rsc": _v(10.76, None, None, 6.31, None, None, None, None, None),
    "cyclooctanone-rsc": _v(7.33, None, None, 9.01, None, None, 37.38, None, None),
    "cyclononanone-rsc": _v(20.43, None, None, 22.57, None, None, 55.28, None, None),
    "cyclodecanone-rsc": _v(15.70, None, None, 17.73, None, None, None, None, None),
    "cycloundecanone-rsc": _v(19.39, None, None, 20.53, None, None, None, None, None),
    "cyclododecanone-rsc": _v(12.91, None, None, 18.02, None, None, 47.11, None, None),
    "cyclopentadecanone-rsc": _v(9.41, None, None, None, None, None, 74.77, None, None),
    "cycloheptadecanone-rsc": _v(4.87, None, None, None, None, None, 89.49, None, None),
    "cyclobutane-1,3-dione-rsc": _v(140.48, None, None, None, None, None, 94.10, None, None),
    "glutaric-anhydride-rsc": _v(20.89, None, None, None, None, None, 8.91, None, None),
    "succinic-anhydride-rsc": _v(4.76, None, None, -11.08, None, None, -10.60, None, None),
    "phthalic-anhydride-rsc": _v(30.66, None, None, None, None, None, -5.52, None, None),
    "ethylene-oxide-rsc": _v(114.62, -10.92, 132.0, 104.82, -23.90, 80.50, None, None, None),
    "trimethylene-oxide-rsc": _v(107.35, None, None, None, -22.38, None, None, None, None),
    "tetrahydrofuran-rsc": _v(24.28, -28.73, 113.66, 17.70, -28.49, 47.18, 14.60, None, None),
    "tetrahydropyran-rsc": _v(5.71, None, None, 1.32, -42.22, None, 0.80, None, None),
    "furan-rsc": _v(-12.18, None, None, -9.97, None, None, None, None, None),
    "1,3-dioxolane-rsc": _v(29.06, None, None, 18.75, -37.74, None, None, None, None),
    "1,3-dioxane-rsc": _v(10.90, None, None, 4.40, -42.26, None, None, None, None),
    "1,4-dioxane-rsc": _v(19.15, -24.34, 73.16, 9.76, -29.50, 86.28, -12.00, None, None),
    "1,3-dioxepane-rsc": _v(25.52, None, None, 20.01, -49.20, None, None, None, None),
    "trioxane-rsc": _v(25.02, None, None, None, None, None, None, None, None),
    "tetraoxane-rsc": _v(34.23, None, None, None, None, None, None, None, None),
    "beta-propiolactone-rsc": _v(97.95, None, None, 75.43, -5.40, 31.85, None, None, None),
    "gamma-butyrolactone-rsc": _v(34.98, None, None, 10.16, -16.61, 21.56, None, None, None),
    "gamma-valerolactone-rsc": _v(26.06, None, None, 4.75, None, None, None, None, None),
    "delta-valerolactone-rsc": _v(42.51, None, None, 19.19, -16.74, 10.77, None, None, None),
    "caprolactone-rsc": _v(None, None, None, None, -21.92, -4.92, None, None, None),
    "undecanolactone-rsc": _v(None, None, None, None, -28.12, -33.05, None, None, None),
    "ethylene-carbonate-rsc": _v(None, None, None, None, None, None, 23.90, None, None),
    "cyclobutane-methyl-carboxylate-rsc": _v(75.21, None, None, 79.08, None, None, None, None, None),
    "bicyclobutane-methyl-carboxylate-rsc": _v(222.27, None, None, 219.98, None, None, None, None, None),
    "1,4-dimethylcubane-dicarboxylate-rsc": _v(595.80, None, None, None, None, None, 590.73, None, None),
    "2-deoxy-D-ribose-rsc": _v(None, None, None, None, None, None, 0.25, None, None),
    "beta-D-ribose-rsc": _v(None, None, None, None, None, None, 12.65, None, None),
    "alpha-D-glucose-rsc": _v(None, None, None, None, None, None, 6.30, None, None),
    "COOH-COOH-ortho": _v(None, None, None, None, None, None, 34.14, 15.00, 8.96),
    "COOH-COOH-meta": _v(-23.94, None, None, None, None, None, 13.14, 30.00, 0.00),
    "CH3O-COOH-ortho": _v(15.00, None, None, None, None, None, 23.00, None, None),
    "CH3O-COOH-meta": _v(5.00, None, None, None, None, None, 5.00, None, None),
    "OH-OH-ortho": _v(7.00, None, None, None, None, None, 16.00, None, None),
    "OH-OH-meta": _v(0.00, None, None, None, None, None, 2.00, None, None),
    "OH-COOH-ortho": _v(-20.00, None, None, None, None, None, 0.00, None, None),
    "zwitterion-aliphatic": _v(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -55.10, -44.50, -13.40),
    "zwitterion-aromatic-I": _v(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -32.00, -20.50, -13.00),
    "zwitterion-aromatic-II": _v(0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -11.00, 5.00, -9.00),
    "ethyleneimine-rsc": _v(115.53, -5.13, 137.90, 101.98, None, None, None, None, None),
    "pyrrolidine-rsc": _v(26.71, -22.29, 118.45, 20.36, -24.48, 42.40, None, None, None),
    "piperidine-rsc": _v(3.14, None, None, -1.09, -29.79, 15.98, None, None, None),
    "hexamethyleneimine-rsc": _v(None, None, None, None, -36.86, None, None, None, None),
    "octahydroazocine-rsc": _v(None, None, None, None, -42.31, None, None, None, None),
    "pyrrolizidine-rsc": _v(35.42, None, None, 18.87, None, None, None, None, None),
    "3,5-dimethylpyrrolizidine-rsc": _v(38.46, None, None, 20.05, None, None, None, None, None),
    "trimethyl-cyanurate-rsc": _v(-95.00, None, None, None, None, None, -120.40, None, None),
    "succinimide-rsc": _v(25.70, None, None, None, None, None, 16.70, None, None),
    "glutarimide-rsc": _v(28.23, None, None, None, None, None, 17.57, None, None),
    "azetidine-rsc": _v(116.00, None, None, 102.00, None, None, None, None, None),
    "pyrrole-rsc": _v(-30.48, None, None, -20.03, None, None, -17.84, None, None),
    "cyclotetramethylenediazene-rsc": _v(12.86, None, None, -4.34, None, None, None, None, None),
    "cyclotrimethylenediazene-rsc": _v(-10.47, None, None, None, None, None, -23.97, None, None),
    "cyclopropanenitrile-rsc": _v(110.56, None, None, 110.76, -28.53, None, None, None, None),
    "cyclobutanenitrile-rsc": _v(91.39, None, None, 98.69, -28.35, None, None, None, None),
    "cyclopentanenitrile-rsc": _v(10.82, None, None, 22.12, -37.27, None, None, None, None),
    "cyclohexanenitrile-rsc": _v(-5.55, None, None, -0.05, -57.29, None, None, None, None),
    "N-nitrosopiperidine-rsc": _v(45.20, None, None, 48.70, None, None, None, None, None),
    "N-nitropiperidine-rsc": _v(-13.91, None, None, -4.11, None, None, 8.48, None, None),
    "R-salt-rsc": _v(None, None, None, None, None, None, 195.30, None, None),
    "RDX-rsc": _v(32.00, None, None, None, None, None, 30.00, None, None),
    "HMX-rsc": _v(17.00, None, None, None, None, None, 32.00, None, None),
    "DINO-PMTA-rsc": _v(None, None, None, None, None, None, 46.70, None, None),
    "cis-azobenzene": _v(48.40, None, None, None, None, None, 49.10, None, None),
    "azidocyclopentane-rsc": _v(29.42, None, None, 27.02, None, None, None, None, None),
    "azidocyclohexane-rsc": _v(-16.45, None, None, -17.95, None, None, None, None, None),
    "NO2-NO2-ortho": _v(44.00, None, None, 45.25, None, None, 40.60, 3.76, None),
    "NO2-NO2-meta": _v(11.00, None, None, 13.50, None, None, 13.50, 5.84, None),
    "NO2-CH3-ortho": _v(2.00, None, None, 2.00, None, None, 4.00, None, None),
    "NO2-CH3-meta": _v(None, None, None, -4.00, None, None, None, None, None),
    "NO2-OH-ortho": _v(10.00, None, None, 16.00, None, None, 13.00, None, None),
    "NO2-OH-meta": _v(6.00, None, None, 0.00, None, None, 0.00, None, None),
    "NO2-NO2-aliphatic-adjacent": _v(20.00, None, None, 20.00, None, None, 20.00, None, None),
    "NO2-COOH-ortho": _v(25.00, None, None, 30.00, None, None, 25.00, 0.00, None),
    "NO2-COOH-meta": _v(14.00, None, None, 16.00, None, None, 14.00, 0.00, None),
    "NH2-NO2-ortho": _v(-4.00, None, None, -4.00, None, None, -4.00, 0.00, None),
    "NH2-NO2-meta": _v(-10.00, None, None, -10.00, None, None, -10.00, None, None),
    "ONO2-ONO2-aliphatic-adjacent": _v(15.10, None, None, 15.90, None, None, 16.00, None, None),
    "Nr-CH3-ortho": _v(-6.30, None, None, -4.00, None, None, None, None, None),
    "Nr-Nr-ortho": _v(85.06, None, None, 83.16, None, None, None, None, None),
    "CH3-CN-cis": _v(-6.00, None, None, -6.00, None, None, None, None, None),
    "NH2-NH2-ortho": _v(None, None, None, None, None, None, -3.00, None, None),
    "NH2-NH2-meta": _v(None, None, None, None, None, None, -10.00, None, None),
    "NH2-COOH-ortho": _v(None, None, None, 12.00, None, None, 14.00, -4.71, None),
    "NH2-COOH-meta": _v(None, None, None, 2.00, None, None, 4.00, -7.22, None),
    "thiacyclopropane-rsc": _v(81.57, -10.76, 122.10, 75.32, None, None, None, None, None),
    "thiacyclobutane-rsc": _v(80.98, -18.00, 112.89, 74.55, -10.54, 40.57, None, None, None),
    "thiacyclopentane-rsc": _v(6.41, -19.34, 97.87, 2.08, -14.19, 31.08, None, None, None),
    "thiacyclohexane-rsc": _v(-2.02, -24.91, 66.85, -5.09, -21.47, 9.12, None, None, None),
    "thiacycloheptane-rsc": _v(20.53, -31.40, 66.35, 13.84, None, None, None, None, None),
    "2,5-dihydrothiophene-rsc": _v(19.13, None, None, 19.96, None, None, None, None, None),
    "thiophene-rsc": _v(-43.54, -1.59, 22.79, None, None, None, None, None, None),
    "2,3-dihydrothiophene-rsc": _v(7.72, None, None, None, None, None, None, None, None),
    "cis-sulfoxide": _v(4.11, -8.03, 5.06, 5.27, 0.00, 0.00, 5.73, 0.00, 0.00),
    "ortho-F-F": _v(20.90, 0.0, 0.0, 25.0, 0.0, 0.0, 25.50, 0.0, 0.0),
    "ortho-Cl-Cl": _v(9.50, 0.0, 0.0, 14.0, 0.0, 0.0, 8.50, 0.0, 0.0),
    "ortho-F-Cl": _v(13.50, 0.0, 0.0, 18.50, 0.0, 0.0, 19.50, 0.0, 0.0),
    "ortho-F-Br": _v(37.25, 0.0, 0.0, 40.60, 0.0, 0.0, 42.50, 0.0, 0.0),
    "ortho-F-I": _v(85.40, 0.0, 0.0, 83.55, 0.0, 0.0, 85.20, 0.0, 0.0),
    "ortho-I-I": _v(7.56, 0.0, 0.0, 6.96, 0.0, 0.0, 5.50, 0.0, 0.0),
    "ortho-alkyl-X": _v(2.51, 0.0, 0.0, 6.30, 0.0, 0.0, 0.0, 0.0, 0.0),
    "cis-Cl-Cl": _v(-4.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "cis-I-I": _v(3.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "cis-CH3-Br": _v(-4.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "meta-I-I": _v(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.08, 0.0, 0.0),
    "meta-COCl-COCl": _v(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.06, 0.0, 0.0),
    "ortho-COCl-COCl": _v(0.0, 0.0, 0.0, 0.0, 10.58, 0.0, None, 0.0, 0.0),
    "ortho-F-CF3": _v(111.00, 0.0, 0.0, 112.00, 0.0, 0.0, 0.0, 0.0, 0.0),
    "meta-F-CF3": _v(2.00, 0.0, 0.0, 6.00, 0.0, 0.0, 0.0, 0.0, 0.0),
    "ortho-F-CH3": _v(-3.30, 0.0, 0.0, -6.00, 0.0, 0.0, 0.0, 0.0, 0.0),
    "ortho-F-F-prime": _v(8.00, 0.0, 0.0, 8.00, 0.0, 0.0, 8.00, 0.0, 0.0),
    "ortho-Cl-Cl-prime": _v(8.00, 0.0, 0.0, 8.00, 0.0, 0.0, 8.00, 0.0, 0.0),
    "meta-F-F": _v(0.0, 0.0, 0.0, 6.00, 0.0, 0.0, 8.50, 0.0, 0.0),
    "meta-Cl-Cl": _v(-5.00, 0.0, 0.0, 10.00, 0.0, 0.0, 4.00, 0.0, 0.0),
    "ortho-Cl-CHO": _v(-6.75, 0.0, 0.0, 8.50, 0.0, 0.0, 0.0, 0.0, 0.0),
    "ortho-F-COOH": _v(20.00, 0.0, 0.0, 0.0, 0.0, 0.0, 20.00, 0.0, 0.0),
    "ortho-Cl-COCl": _v(0.0, 0.0, 0.0, 34.43, 0.0, 0.0, 0.0, 0.0, 0.0),
    "ortho-F-OH": _v(25.50, 0.0, 0.0, 23.00, 0.0, 0.0, 20.00, 0.0, 0.0),
    "ortho-Cl-COOH": _v(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.00, 0.0, 0.0),
    "ortho-Br-COOH": _v(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.00, 0.0, 0.0),
    "ortho-I-COOH": _v(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.00, 0.0, 0.0),
    "ortho-OH-Cl": _v(7.50, 0.0, 0.0, 0.0, 0.0, 0.0, 11.00, 0.0, 0.0),
    "cis-CH3-I": _v(-4.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


# Table 2 repeats the NH2--NH2 ortho/meta labels on p. 826 with values
# incompatible with both the earlier p. 823 rows and the calculated columns
# for the three phenylenediamines in Table 26.  Preserve those printed cells
# for a complete transcription, but do not expose them as operational
# corrections: a SMILES structure cannot distinguish two identically labelled
# rows, and Table 26 establishes the p. 823 variants used by the method.
AMBIGUOUS_PAPER_CORRECTION_VALUES: dict[str, Values] = {
    "p826-ortho-NH2-NH2": _v(
        -10.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "p826-meta-NH2-NH2": _v(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.00, 0.0, 0.0),
}


# Corrections whose definition is a complete named skeleton rather than a
# local environment.  Keys are canonicalized once at import.  Simple
# monocyclic corrections are generated algorithmically below.
_NAMED_CORRECTION_SMILES = {
    "C1C=CC=C1": "1,3-cyclopentadiene-rsc",
    "C1CC=CC=C1": "1,3-cyclohexadiene-rsc",
    "C1C=CCC=C1": "1,4-cyclohexadiene-rsc",
    "C1CC=CC=CC1": "1,3-cycloheptadiene-rsc",
    "C1CC=CCCC=C1": "1,5-cyclooctadiene-rsc",
    "C1C=CC=CC=C1": "1,3,5-cycloheptatriene-rsc",
    "C1=C\\C=C/C=C\\C=C/1": "cyclooctatetraene-rsc",
    "C1CC12CC2": "spiropentane-rsc",
    "C1C2CC3CC1CC(C2)C3": "adamantane-rsc",
    "c1ccc2c(c1)c1cccc3cccc2c13": "fluoranthene-rsc",
    "C1CC2CCC1CC2": "bicyclo[2.2.2]octane-rsc",
    "C1CC2CCCC(C1)CCC2": "bicyclo[3.3.3]undecane-rsc",
    "C1CC[C@H]2CCCC[C@H]2C1": "cis-decalin-rsc",
    "C1CC[C@H]2CCCC[C@@H]2C1": "trans-decalin-rsc",
    "C1CC[C@H]2CCC[C@H]2C1": "cis-hexahydroindan-rsc",
    "C1CC[C@H]2CCC[C@@H]2C1": "trans-hexahydroindan-rsc",
    "C1CCC[C@H]2C[C@H]2CC1": "cis-bicyclo[6.1.0]nonane-rsc",
    "C1CCC[C@H]2C[C@@H]2CC1": "trans-bicyclo[6.1.0]nonane-rsc",
    "C1C2C1C2": "bicyclo[1.1.0]butane-rsc",
    "C1C2C3C2C4C1C34": "tetracyclo[3.2.0.2,7.0.4,6]heptane-rsc",
    "C1C2CC3C1C3C2": "tricyclo[2.2.1.0.2,6]heptane-rsc",
    "C12C3C4C1C5C2C3C45": "pentacyclo[4.2.0.2,5.0.3,8.0.4,7]octane-rsc",
    "C1CC2CC2C1": "bicyclo[3.1.0]hexane-rsc",
    "C1C2C=CC1C=C2": "bicyclo[2.2.1]hepta-2,5-diene-rsc",
    "C1CC2CC1C=C2": "bicyclo[2.2.1]hept-2-ene-rsc",
    "C1CC2CCC1C2": "bicyclo[2.2.1]heptane-rsc",
    "C1CCC2CC2C1": "bicyclo[4.1.0]heptane-rsc",
    "C1CC2CCC1C=C2": "bicyclo[2.2.2]oct-2-ene-rsc",
    "C1CCC2CCC2C1": "bicyclo[4.2.0]octane-rsc",
    "C1CCC2CC2CC1": "bicyclo[5.1.0]octane-rsc",
    "C1CC2CCCC(C1)C2": "bicyclo[3.3.1]nonane-rsc",
    "C1CC2CCCC2C1": "cis-bicyclo[3.3.0]octane-rsc",
    "C1C[C@H]2CCC[C@H]2C1": "cis-bicyclo[3.3.0]octane-rsc",
    "C1C[C@H]2CCC[C@@H]2C1": "trans-bicyclo[3.3.0]octane-rsc",
    "C1CC2=CC(=CC=C2)CCC3=CC=CC1=C3": "2,2-metacyclophane-rsc",
    "C1CC2=CC(=CC=C2)CCC3=CC=C1C=C3": "2,2-metaparacyclophane-rsc",
    "C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2": "2,2-paracyclophane-rsc",
    "C1CC2=CC=C(CCCC3=CC=C(C1)C=C3)C=C2": "3,3-paracyclophane-rsc",
    "O=C1CCC(=O)1": "cyclobutane-1,3-dione-rsc",
    "O=C1CCC(=O)O1": "succinic-anhydride-rsc",
    "O=C1CCCC(=O)O1": "glutaric-anhydride-rsc",
    "O=c1oc(=O)c2ccccc12": "phthalic-anhydride-rsc",
    "C1=COC=C1": "furan-rsc",
    "C1COCO1": "1,3-dioxolane-rsc",
    "C1COCOC1": "1,3-dioxane-rsc",
    "C1COCCO1": "1,4-dioxane-rsc",
    "C1CCOCOC1": "1,3-dioxepane-rsc",
    "C1OCOCO1": "trioxane-rsc",
    "C1OCOCOCO1": "tetraoxane-rsc",
    "O=C1OCC1": "beta-propiolactone-rsc",
    "O=C1OCCC1": "gamma-butyrolactone-rsc",
    "CC1CCC(=O)O1": "gamma-valerolactone-rsc",
    "O=C1OCCCC1": "delta-valerolactone-rsc",
    "O=C1OCCCCC1": "caprolactone-rsc",
    "CCCCCCC1CCCC(=O)O1": "undecanolactone-rsc",
    "O=C1OCCO1": "ethylene-carbonate-rsc",
    "COC(=O)C1CCC1": "cyclobutane-methyl-carboxylate-rsc",
    "COC(=O)C12CC1C2": "bicyclobutane-methyl-carboxylate-rsc",
    "COC(=O)C12C3C4C1C5C2C3C45C(=O)OC": "1,4-dimethylcubane-dicarboxylate-rsc",
    "O=CC[C@H](O)[C@H](O)CO": "2-deoxy-D-ribose-rsc",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@H]1O": "beta-D-ribose-rsc",
    "OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@H]1O": "alpha-D-glucose-rsc",
    "N1CC1": "ethyleneimine-rsc",
    "N1CCCC1": "pyrrolidine-rsc",
    "N1CCCCC1": "piperidine-rsc",
    "N1CCCCCC1": "hexamethyleneimine-rsc",
    "N1CCCCCCC1": "octahydroazocine-rsc",
    "C1CC2CCCN2C1": "pyrrolizidine-rsc",
    "CC1CCC2CCC(C)N12": "3,5-dimethylpyrrolizidine-rsc",
    "CN1C(=O)N(C)C(=O)N(C)C1=O": "trimethyl-cyanurate-rsc",
    "O=C1CCC(=O)N1": "succinimide-rsc",
    "O=C1CCCC(=O)N1": "glutarimide-rsc",
    "N1CCC1": "azetidine-rsc",
    "c1cc[nH]c1": "pyrrole-rsc",
    "C1CCCN=N1": "cyclotetramethylenediazene-rsc",
    "C1CCN=N1": "cyclotrimethylenediazene-rsc",
    "N#CC1CC1": "cyclopropanenitrile-rsc",
    "N#CC1CCC1": "cyclobutanenitrile-rsc",
    "N#CC1CCCC1": "cyclopentanenitrile-rsc",
    "N#CC1CCCCC1": "cyclohexanenitrile-rsc",
    "O=NN1CCCCC1": "N-nitrosopiperidine-rsc",
    "O=[N+]([O-])N1CCCCC1": "N-nitropiperidine-rsc",
    "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "RDX-rsc",
    "C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "HMX-rsc",
    "C1N(CN(CN1N=O)N=O)N=O": "R-salt-rsc",
    "C1N2CN(CN1CN(C2)N=O)N=O": "DINO-PMTA-rsc",
    "c1ccccc1/N=N\\c1ccccc1": "cis-azobenzene",
    "[N-]=[N+]=NC1CCCC1": "azidocyclopentane-rsc",
    "[N-]=[N+]=NC1CCCCC1": "azidocyclohexane-rsc",
    "C1C=CCS1": "2,5-dihydrothiophene-rsc",
    "c1ccsc1": "thiophene-rsc",
    "C1CSC=C1": "2,3-dihydrothiophene-rsc",
}


def _build_named_corrections() -> dict[str, str]:
    result = {}
    for smiles, correction in _NAMED_CORRECTION_SMILES.items():
        mol = Chem.MolFromSmiles(smiles.replace(" ", ""))
        if mol is not None:
            result[Chem.MolToSmiles(mol, isomericSmiles=True)] = correction
    return result


_NAMED_STRUCTURE_CORRECTIONS = _build_named_corrections()

_BIPHENANTHRENE_CANONICAL = Chem.MolToSmiles(Chem.MolFromSmiles(
    "C1=CC=C2C(=C1)C=C(C3=CC=CC=C23)C4=CC5=CC=CC=C5C6=CC=CC=C64"))


@dataclass
class Fragmentation:
    smiles: str
    mol: Chem.Mol
    groups: Counter[str] = field(default_factory=Counter)
    corrections: Counter[str] = field(default_factory=Counter)
    assignments: dict[str, tuple[int, ...]] = field(default_factory=dict)
    symmetry_number: int = 1
    optical_isomers: int | None = 1


@dataclass(frozen=True)
class PhaseResult:
    enthalpy_formation_kJ_mol: float | None
    heat_capacity_J_mol_K: float | None
    entropy_J_mol_K: float | None
    entropy_formation_J_mol_K: float | None
    gibbs_formation_kJ_mol: float | None
    ln_formation_equilibrium_constant: float | None


@dataclass(frozen=True)
class DomalskiHearingResult:
    smiles: str
    temperature_K: float
    pressure_Pa: float
    gas: PhaseResult
    liquid: PhaseResult
    solid: PhaseResult
    groups: dict[str, int]
    corrections: dict[str, int]
    symmetry_number: int
    optical_isomers: int | None
    warnings: tuple[str, ...]

    def summary(self) -> str:
        lines = [
            f"SMILES: {self.smiles}",
            f"groups: {self.groups}",
            f"corrections: {self.corrections or {}}",
            f"symmetry: {self.symmetry_number}; optical isomers: {self.optical_isomers}",
        ]
        for name, phase in (("gas", self.gas), ("liquid", self.liquid), ("solid", self.solid)):
            lines.append(
                f"{name}: Hf={phase.enthalpy_formation_kJ_mol!r} kJ/mol, "
                f"Cp={phase.heat_capacity_J_mol_K!r} J/(mol K), "
                f"S={phase.entropy_J_mol_K!r} J/(mol K), "
                f"Gf={phase.gibbs_formation_kJ_mol!r} kJ/mol")
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class HfApplicability:
    """Structure-only applicability assessment for gas Hf estimation."""

    applicable: bool
    reasons: tuple[str, ...] = ()


def _atom_kind(atom: Chem.Atom) -> str:
    if atom.GetIsAromatic():
        mol = atom.GetOwningMol()
        aromatic_ring_count = sum(
            atom.GetIdx() in ring
            and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            for ring in mol.GetRingInfo().AtomRings())
        return "CBF" if aromatic_ring_count > 1 else "CB"
    if atom.GetAtomicNum() != 6:
        return atom.GetSymbol()
    hybrid = atom.GetHybridization()
    if hybrid == Chem.HybridizationType.SP:
        return "Ct"
    if hybrid == Chem.HybridizationType.SP2:
        return "Cd"
    return "C"


def _format_group(center: str, tokens: list[str]) -> str:
    order = {"H": 0, "C": 1, "Cd": 2, "Ct": 3, "CB": 4, "CO": 5,
             "CBF": 5, "CN": 6, "N": 7, "NO2": 8, "N3": 9, "O": 10, "S": 11,
             "SO": 12, "SO2": 13, "F": 14, "Cl": 15, "Br": 16, "I": 17}
    counts = Counter(tokens)
    parts = []
    for token in sorted(counts, key=lambda item: (order.get(item, 99), item)):
        count = counts[token]
        parts.append(f"({token})" + (str(count) if count > 1 else ""))
    return center + "-" + "".join(parts)


def _carbonyl_oxygen(atom: Chem.Atom) -> Chem.Atom | None:
    if atom.GetAtomicNum() != 6:
        return None
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
            return other
    return None


def _is_carbonyl(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and _carbonyl_oxygen(atom) is not None


def _sulfur_oxide_count(atom: Chem.Atom) -> int:
    if atom.GetAtomicNum() != 16:
        return 0
    return sum(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds()
    )


def _functional_token(atom: Chem.Atom) -> str:
    if _is_carbonyl(atom):
        return "CO"
    oxides = _sulfur_oxide_count(atom)
    if oxides:
        return "SO2" if oxides == 2 else "SO"
    return _atom_kind(atom)


def _nitro_atoms(nitrogen: Chem.Atom) -> tuple[int, ...] | None:
    if nitrogen.GetAtomicNum() != 7:
        return None
    oxygens = [nb for nb in nitrogen.GetNeighbors() if nb.GetAtomicNum() == 8]
    if len(oxygens) == 2 and nitrogen.GetDegree() == 3:
        return (nitrogen.GetIdx(), *(atom.GetIdx() for atom in oxygens))
    return None


def _carboxylic_acid_carbons(mol: Chem.Mol) -> list[Chem.Atom]:
    result = []
    for atom in mol.GetAtoms():
        oxygen = _carbonyl_oxygen(atom)
        if oxygen is None:
            continue
        if any(nb.GetAtomicNum() == 8 and nb.GetIdx() != oxygen.GetIdx()
               and nb.GetTotalNumHs() for nb in atom.GetNeighbors()):
            result.append(atom)
    return result


def _primary_amine_nitrogens(mol: Chem.Mol) -> list[Chem.Atom]:
    return [atom for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 7 and not atom.GetIsAromatic()
            and atom.GetTotalNumHs() == 2 and atom.GetDegree() == 1]


_EXOCYCLIC_CARBONYL_AROMATIC = Chem.MolFromSmarts("[c](=[O,S])")


def _demote_exocyclic_carbonyl_rings(mol: Chem.Mol) -> Chem.Mol:
    """Restore Kekule atom types in lactam/lactone-style aromatic rings."""
    carbonyls = {
        match[0] for match in mol.GetSubstructMatches(
            _EXOCYCLIC_CARBONYL_AROMATIC)
    }
    if not carbonyls:
        return mol
    result = Chem.Mol(mol)
    try:
        Chem.Kekulize(result, clearAromaticFlags=False)
    except Exception:
        return mol
    rings = result.GetRingInfo().AtomRings()
    affected = [set(ring) for ring in rings if carbonyls & set(ring)]
    unaffected_aromatic = set().union(*(
        set(ring) for ring in rings
        if not (carbonyls & set(ring))
        and all(result.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )) if any(not (carbonyls & set(ring)) for ring in rings) else set()
    demote = set().union(*affected) - unaffected_aromatic
    for idx in demote:
        result.GetAtomWithIdx(idx).SetIsAromatic(False)
    for bond in result.GetBonds():
        if bond.GetBeginAtomIdx() in demote or bond.GetEndAtomIdx() in demote:
            bond.SetIsAromatic(False)
    return result


def _resolve_group(key: str, atom: Chem.Atom, oxygen_context: str | None = None) -> str:
    if key in GROUP_VALUES:
        return key
    if key.startswith("C-(H)3("):
        token = key[len("C-(H)3("):-1]
        alias = f"C-(H)3({token})"
        if alias in GROUP_VALUES:
            return alias
        if token in {"Cd", "Ct", "CB", "CO", "N", "NA", "N1", "O", "S", "SO", "SO2"}:
            return "C-(H)3(C)"
    if key == "C-(H)(C)2(O)":
        suffix = "alcohol" if oxygen_context == "alcohol" else "ether"
        return key + "-" + suffix
    if key == "C-(C)3(O)":
        suffix = "alcohol" if oxygen_context == "alcohol" else "ether"
        return key + "-" + suffix
    if key.startswith("CB-"):
        tokens = []
        for token, count in re.findall(r"\(([^)]+)\)(\d*)", key):
            tokens.extend([token] * (int(count) if count else 1))
        framework = [token for token in tokens if token == "CB"]
        substituents = [token for token in tokens if token != "CB"]
        reordered = _format_group("CB", substituents + framework)
        # _format_group's global order still puts CB first; aromatic notation
        # conventionally prints its exocyclic substituent before (CB)2.
        if len(framework) == 2 and len(substituents) == 1:
            reordered = f"CB-({substituents[0]})(CB)2"
        if reordered in GROUP_VALUES:
            return reordered
    if key.startswith("CBF-"):
        tokens = []
        for token, count in re.findall(r"\(([^)]+)\)(\d*)", key):
            tokens.extend([token] * (int(count) if count else 1))
        candidates = []
        if Counter(tokens) == Counter({"CB": 2, "CBF": 1}):
            candidates.append("CBF-(CBF)(CB)2")
        elif Counter(tokens) == Counter({"CB": 1, "CBF": 2}):
            candidates.append("CBF-(CB)(CBF)2")
        elif Counter(tokens) == Counter({"CBF": 3}):
            candidates.append("CBF-(CBF)3")
        for candidate in candidates:
            if candidate in GROUP_VALUES:
                return candidate
    if key == "S-(H)(C)":
        return "S-(C)(H)"
    if key == "S-(H)(CB)":
        return "S-(CB)(H)"
    if key == "C-(H)(F)(Cl)(Br)":
        return "C-(Br)(Cl)(F)"
    if key == "Cd-(H)(Ct)":
        # Table 2's group identities (p. 826) equate Cd-(H)(Ct) with
        # Cd-(H)(Cd), not with the aliphatic-neighbour Cd-(H)(C) row.
        return "Cd-(H)(Cd)"
    if key == "Cd-(C)(Ct)":
        return "Cd-(C)2"
    center = key.split("-", 1)[0]
    tokens = Counter()
    for token, count in re.findall(r"\(([^)]+)\)(\d*)", key):
        tokens[token] += int(count) if count else 1
    permutation_matches = []
    for candidate in GROUP_VALUES:
        if candidate.split("-", 1)[0] != center:
            continue
        candidate_tokens = Counter()
        for token, count in re.findall(r"\(([^)]+)\)(\d*)", candidate):
            candidate_tokens[token] += int(count) if count else 1
        if candidate_tokens == tokens:
            permutation_matches.append(candidate)
    if len(permutation_matches) == 1:
        return permutation_matches[0]
    raise FragmentationError(
        f"no published Domalski--Hearing group for atom {atom.GetIdx()} "
        f"({atom.GetSymbol()}) in environment {key}")


def _oxygen_context(atom: Chem.Atom) -> str | None:
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        if nb.GetTotalNumHs() and nb.GetDegree() == 1:
            return "alcohol"
        if any(other.GetAtomicNum() == 8 for other in nb.GetNeighbors()
               if other.GetIdx() != atom.GetIdx()):
            return "alcohol"
    return "ether"


def _aromatic_substituent_label(host: Chem.Atom, substituent: Chem.Atom) -> str:
    """Table-2 label for one exocyclic aromatic substituent."""
    z = substituent.GetAtomicNum()
    if z in (9, 17, 35, 53):
        return substituent.GetSymbol()
    if z == 7:
        if _nitro_atoms(substituent):
            return "NO2"
        if substituent.GetTotalNumHs() == 2:
            return "NH2"
        return "N"
    if z == 8:
        if substituent.GetTotalNumHs():
            return "OH"
        other = [nb for nb in substituent.GetNeighbors() if nb.GetIdx() != host.GetIdx()]
        if len(other) == 1 and other[0].GetAtomicNum() == 6 and other[0].GetTotalNumHs() == 3:
            return "CH3O"
        return "O"
    if z == 6:
        if _is_carbonyl(substituent):
            neighbors = [nb for nb in substituent.GetNeighbors()
                         if nb.GetIdx() != host.GetIdx()
                         and nb.GetIdx() != _carbonyl_oxygen(substituent).GetIdx()]
            if any(nb.GetAtomicNum() == 8 and nb.GetTotalNumHs() for nb in neighbors):
                return "COOH"
            if any(nb.GetAtomicNum() == 17 for nb in neighbors):
                return "COCl"
            if substituent.GetTotalNumHs():
                return "CHO"
            return "CO"
        halogens = [nb.GetSymbol() for nb in substituent.GetNeighbors()
                    if nb.GetAtomicNum() in (9, 17, 35, 53)]
        if halogens.count("F") == 3:
            return "CF3"
        if substituent.GetTotalNumHs() == 3:
            return "CH3"
        return "alkyl"
    return substituent.GetSymbol()


_AROMATIC_PAIR_CORRECTIONS = {
    (1, frozenset(("alkyl", "alkyl"))): "ortho-hydrocarbon",
    (1, frozenset(("CH3", "CH3"))): "ortho-hydrocarbon",
    (2, frozenset(("alkyl", "alkyl"))): "meta-hydrocarbon",
    (2, frozenset(("CH3", "CH3"))): "meta-hydrocarbon",
    (1, frozenset(("COOH", "COOH"))): "COOH-COOH-ortho",
    (2, frozenset(("COOH", "COOH"))): "COOH-COOH-meta",
    (1, frozenset(("CH3O", "COOH"))): "CH3O-COOH-ortho",
    (2, frozenset(("CH3O", "COOH"))): "CH3O-COOH-meta",
    (1, frozenset(("OH", "OH"))): "OH-OH-ortho",
    (2, frozenset(("OH", "OH"))): "OH-OH-meta",
    (1, frozenset(("OH", "COOH"))): "OH-COOH-ortho",
    (1, frozenset(("NO2", "NO2"))): "NO2-NO2-ortho",
    (2, frozenset(("NO2", "NO2"))): "NO2-NO2-meta",
    (1, frozenset(("NO2", "CH3"))): "NO2-CH3-ortho",
    (2, frozenset(("NO2", "CH3"))): "NO2-CH3-meta",
    (1, frozenset(("NO2", "OH"))): "NO2-OH-ortho",
    (2, frozenset(("NO2", "OH"))): "NO2-OH-meta",
    (1, frozenset(("NO2", "COOH"))): "NO2-COOH-ortho",
    (2, frozenset(("NO2", "COOH"))): "NO2-COOH-meta",
    (1, frozenset(("NH2", "NO2"))): "NH2-NO2-ortho",
    (2, frozenset(("NH2", "NO2"))): "NH2-NO2-meta",
    (1, frozenset(("NH2", "NH2"))): "NH2-NH2-ortho",
    (2, frozenset(("NH2", "NH2"))): "NH2-NH2-meta",
    (1, frozenset(("NH2", "COOH"))): "NH2-COOH-ortho",
    (2, frozenset(("NH2", "COOH"))): "NH2-COOH-meta",
    (1, frozenset(("F", "F"))): "ortho-F-F",
    (2, frozenset(("F", "F"))): "meta-F-F",
    (1, frozenset(("Cl", "Cl"))): "ortho-Cl-Cl",
    (2, frozenset(("Cl", "Cl"))): "meta-Cl-Cl",
    (1, frozenset(("I", "I"))): "ortho-I-I",
    (2, frozenset(("I", "I"))): "meta-I-I",
    (1, frozenset(("F", "Cl"))): "ortho-F-Cl",
    (1, frozenset(("F", "Br"))): "ortho-F-Br",
    (1, frozenset(("F", "I"))): "ortho-F-I",
    (1, frozenset(("F", "CF3"))): "ortho-F-CF3",
    (2, frozenset(("F", "CF3"))): "meta-F-CF3",
    (1, frozenset(("F", "CH3"))): "ortho-F-CH3",
    (1, frozenset(("Cl", "CHO"))): "ortho-Cl-CHO",
    (1, frozenset(("F", "COOH"))): "ortho-F-COOH",
    (1, frozenset(("Cl", "COCl"))): "ortho-Cl-COCl",
    (1, frozenset(("COCl", "COCl"))): "ortho-COCl-COCl",
    (2, frozenset(("COCl", "COCl"))): "meta-COCl-COCl",
    (1, frozenset(("F", "OH"))): "ortho-F-OH",
    (1, frozenset(("Cl", "COOH"))): "ortho-Cl-COOH",
    (1, frozenset(("Br", "COOH"))): "ortho-Br-COOH",
    (1, frozenset(("I", "COOH"))): "ortho-I-COOH",
    (1, frozenset(("OH", "Cl"))): "ortho-OH-Cl",
}


def _assign_groups(smiles: str, mol: Chem.Mol) -> Fragmentation:
    canonical_for_corrections = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = _demote_exocyclic_carbonyl_rings(mol)
    frag = Fragmentation(smiles=smiles, mol=mol)
    claimed: set[int] = set()
    token_overrides: dict[tuple[int, int], str] = {}
    amino_acid = bool(_carboxylic_acid_carbons(mol) and _primary_amine_nitrogens(mol))
    primary_amines_seen = 0
    azo_nitrogens: set[int] = set()
    azo_oxide_nitrogens: set[int] = set()
    imine_nitrogens: set[int] = set()
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if left.GetAtomicNum() == right.GetAtomicNum() == 7:
            azo_nitrogens.update((left.GetIdx(), right.GetIdx()))
            for nitrogen in (left, right):
                oxide_oxygens = [nb for nb in nitrogen.GetNeighbors()
                                 if nb.GetAtomicNum() == 8]
                if oxide_oxygens:
                    azo_oxide_nitrogens.add(nitrogen.GetIdx())
                    claimed.update(nb.GetIdx() for nb in oxide_oxygens)
        elif {left.GetAtomicNum(), right.GetAtomicNum()} == {6, 7}:
            nitrogen = left if left.GetAtomicNum() == 7 else right
            if not nitrogen.GetIsAromatic():
                imine_nitrogens.add(nitrogen.GetIdx())
    for idx in azo_nitrogens:
        nitrogen = mol.GetAtomWithIdx(idx)
        for host in nitrogen.GetNeighbors():
            if host.GetAtomicNum() != 7 and host.GetAtomicNum() != 8:
                token_overrides[(host.GetIdx(), idx)] = "NA"

    # Nitrile, nitro, azide, and halogen atoms are included in the group on
    # the atom to which the functional unit is attached (the convention in
    # Table 2), rather than receiving independent atom-centred groups.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            for bond in atom.GetBonds():
                nb = bond.GetOtherAtom(atom)
                if bond.GetBondType() == Chem.BondType.TRIPLE and nb.GetAtomicNum() == 7:
                    hosts = [x for x in atom.GetNeighbors() if x.GetIdx() != nb.GetIdx()]
                    if len(hosts) != 1:
                        raise FragmentationError("terminal nitrile without one organic host")
                    host = hosts[0]
                    oxide_oxygens = [x for x in nb.GetNeighbors()
                                     if x.GetAtomicNum() == 8]
                    token_overrides[(host.GetIdx(), atom.GetIdx())] = (
                        "CNO" if oxide_oxygens else "CN")
                    claimed.update((atom.GetIdx(), nb.GetIdx()))
                    claimed.update(x.GetIdx() for x in oxide_oxygens)
        if atom.GetAtomicNum() == 7:
            oxygens = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8]
            bridging = [oxygen for oxygen in oxygens
                        if any(nb.GetAtomicNum() == 6 for nb in oxygen.GetNeighbors())]
            if len(bridging) == 1 and len(oxygens) in (2, 3):
                bridge = bridging[0]
                token_overrides[(bridge.GetIdx(), atom.GetIdx())] = (
                    "NO2" if len(oxygens) == 3 else "NO")
                claimed.add(atom.GetIdx())
                claimed.update(oxygen.GetIdx() for oxygen in oxygens
                               if oxygen.GetIdx() != bridge.GetIdx())
                continue
            nitrogens = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7]
            hosts = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 7]
            if len(nitrogens) == 1 and len(hosts) == 1 and hosts[0].GetAtomicNum() != 8:
                component = {atom.GetIdx()}
                frontier = [atom]
                while frontier:
                    current = frontier.pop()
                    for nb in current.GetNeighbors():
                        if nb.GetAtomicNum() == 7 and nb.GetIdx() not in component:
                            component.add(nb.GetIdx())
                            frontier.append(nb)
                if len(component) == 3:
                    host = hosts[0]
                    token_overrides[(host.GetIdx(), atom.GetIdx())] = "N3"
                    claimed.update(component)
                    continue
            if len(oxygens) == 1 and atom.GetDegree() == 2:
                host = next(nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 8)
                token_overrides[(host.GetIdx(), atom.GetIdx())] = "NO"
                claimed.update((atom.GetIdx(), oxygens[0].GetIdx()))
                continue
        nitro = _nitro_atoms(atom)
        if nitro:
            hosts = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 8]
            if len(hosts) != 1:
                raise FragmentationError("nitro group without one organic host")
            host = hosts[0]
            token_overrides[(host.GetIdx(), atom.GetIdx())] = "NO2"
            claimed.update(nitro)
        if atom.GetAtomicNum() in (9, 17, 35, 53):
            if atom.GetDegree() != 1:
                raise FragmentationError("a halogen must have exactly one host atom")
            host = atom.GetNeighbors()[0]
            token_overrides[(host.GetIdx(), atom.GetIdx())] = atom.GetSymbol()
            claimed.add(atom.GetIdx())

    # Carbonyl groups claim the carbon and double-bonded oxygen together.
    for atom in mol.GetAtoms():
        oxygen = _carbonyl_oxygen(atom)
        if oxygen is None:
            continue
        tokens = ["H"] * atom.GetTotalNumHs()
        for nb in atom.GetNeighbors():
            if nb.GetIdx() == oxygen.GetIdx():
                continue
            if nb.GetIsAromatic():
                tokens.append("CB")
            else:
                tokens.append(token_overrides.get(
                    (atom.GetIdx(), nb.GetIdx()), _functional_token(nb)))
        key = _format_group("CO", tokens)
        if key == "CO-(CO)(O)":
            single_oxygen = next(
                (nb for nb in atom.GetNeighbors()
                 if nb.GetAtomicNum() == 8 and nb.GetIdx() != oxygen.GetIdx()),
                None)
            if single_oxygen is not None and not single_oxygen.GetTotalNumHs():
                key = "CO-(O)(CO)"
        key = _resolve_group(key, atom)
        if amino_acid and key == "CO-(CB)(N)":
            key += "-amino-acid"
        frag.groups[key] += 1
        frag.assignments[f"{key}@{atom.GetIdx()}"] = (atom.GetIdx(), oxygen.GetIdx())
        claimed.update((atom.GetIdx(), oxygen.GetIdx()))

    # Sulfoxide/sulfone oxygen atoms belong to the S-centred group.
    for atom in mol.GetAtoms():
        oxides = _sulfur_oxide_count(atom)
        if not oxides:
            continue
        center = "SO2" if oxides == 2 else "SO"
        tokens = []
        oxygen_indices = []
        for bond in atom.GetBonds():
            nb = bond.GetOtherAtom(atom)
            if bond.GetBondType() == Chem.BondType.DOUBLE and nb.GetAtomicNum() == 8:
                oxygen_indices.append(nb.GetIdx())
            else:
                tokens.append(_functional_token(nb))
        key = _format_group(center, tokens)
        key = _resolve_group(key, atom)
        frag.groups[key] += 1
        frag.assignments[f"{key}@{atom.GetIdx()}"] = (atom.GetIdx(), *oxygen_indices)
        claimed.update((atom.GetIdx(), *oxygen_indices))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in claimed:
            continue
        z = atom.GetAtomicNum()
        tokens = ["H"] * atom.GetTotalNumHs()
        if z == 6:
            kind = _atom_kind(atom)
            if (atom.GetDegree() == 2 and all(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    for bond in atom.GetBonds())):
                key = "Ca"
            elif atom.GetIsAromatic() and sum(
                    nb.GetAtomicNum() == 7 and nb.GetIsAromatic()
                    for nb in atom.GetNeighbors()) == 2:
                key = "Cd-(H)(N1)2"
            else:
                for bond in atom.GetBonds():
                    nb = bond.GetOtherAtom(atom)
                    if ((kind == "Cd" and bond.GetBondType() == Chem.BondType.DOUBLE
                         and nb.GetAtomicNum() in (6, 7))
                            or (kind == "Ct" and bond.GetBondType() == Chem.BondType.TRIPLE
                                and nb.GetAtomicNum() == 6)):
                        continue
                    token = token_overrides.get((idx, nb.GetIdx()))
                    if token is not None:
                        tokens.append(token)
                    elif kind == "CB" and nb.GetIsAromatic():
                        tokens.append(
                            _atom_kind(nb) if atom.GetTotalNumHs() == 0
                            else "CB")
                    elif kind == "CBF" and nb.GetIsAromatic():
                        tokens.append(_atom_kind(nb))
                    else:
                        tokens.append(_functional_token(nb))
                key = _format_group(kind, tokens)
        elif z == 8:
            for nb in atom.GetNeighbors():
                tokens.append(token_overrides.get(
                    (idx, nb.GetIdx()), _functional_token(nb)))
            key = _format_group("O", tokens)
            if key == "O-(CO)2":
                carbonyls = [nb for nb in atom.GetNeighbors() if _is_carbonyl(nb)]
                aromatic = any(
                    any(x.GetIsAromatic() for x in carbonyl.GetNeighbors()
                        if x.GetIdx() != atom.GetIdx() and x.GetAtomicNum() == 6)
                    for carbonyl in carbonyls)
                key += "-aromatic" if aromatic else "-aliphatic"
        elif z == 7:
            if idx in azo_nitrogens:
                hosts = [nb for nb in atom.GetNeighbors()
                         if nb.GetAtomicNum() not in (7, 8)]
                if len(hosts) != 1:
                    raise FragmentationError("azo nitrogen without one carbon host")
                host_kind = "CB" if hosts[0].GetIsAromatic() else "C"
                key = f"NA-({'oxide)(' if idx in azo_oxide_nitrogens else ''}{host_kind})"
                if idx in azo_oxide_nitrogens:
                    key = f"NA-(oxide)({host_kind})"
            elif idx in imine_nitrogens:
                hosts = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6
                         and mol.GetBondBetweenAtoms(idx, nb.GetIdx()).GetBondType()
                         != Chem.BondType.DOUBLE]
                key = "Nr-(CB)" if hosts and hosts[0].GetIsAromatic() else "Nr-(C)"
            elif atom.GetIsAromatic():
                key = "N-(H)(CB)2" if atom.GetTotalNumHs() else "Nr-(CB)"
            else:
                for nb in atom.GetNeighbors():
                    token = token_overrides.get(
                        (idx, nb.GetIdx()), _functional_token(nb))
                    tokens.append(token)
                if atom.GetTotalNumHs() == 2 and tokens == ["H", "H", "Cd"]:
                    tokens[-1] = "C"
                elif atom.GetTotalNumHs() == 1 and tokens.count("Cd") == 1:
                    tokens[tokens.index("Cd")] = "C"
                key = _format_group("N", tokens)
        elif z == 16:
            for nb in atom.GetNeighbors():
                tokens.append(token_overrides.get(
                    (idx, nb.GetIdx()), _functional_token(nb)))
            key = _format_group("S", tokens)
        else:
            raise FragmentationError(
                f"atom {idx} ({atom.GetSymbol()}) is not covered by Table 2")
        if key == "C-(H)(C)(O)2":
            key = "C-(H)(C)(O)2" if atom.IsInRing() else "C-(H)(O)2(C)"
        key = _resolve_group(key, atom, _oxygen_context(atom) if z == 6 else None)
        if amino_acid:
            if key == "N-(H)2(C)":
                primary_amines_seen += 1
                if primary_amines_seen > 1:
                    key = "N-(H)2(C)-second-amino-acid"
            elif key in {"N-(H)2(CO)", "N-(H)(C)(CO)", "CO-(CB)(N)"}:
                key += "-amino-acid"
        frag.groups[key] += 1
        frag.assignments[f"{key}@{idx}"] = (idx,)
        claimed.add(idx)

    if len(claimed) != mol.GetNumHeavyAtoms():
        missing = sorted(set(range(mol.GetNumAtoms())) - claimed)
        raise FragmentationError(f"fragmenter left atom indices unclaimed: {missing}")
    if canonical_for_corrections == _BIPHENANTHRENE_CANONICAL:
        # Table 14 explicitly assigns all eight fused junction atoms to
        # CBF-(CBF)(CB)2 for 9,9'-biphenanthrene.  The inter-unit bond is not
        # represented by RDKit as an aromatic ring bond, so local ring flags
        # alone otherwise leave four atoms in the neighbouring CBF subtype.
        moved = frag.groups.pop("CBF-(CB)(CBF)2", 0)
        frag.groups["CBF-(CBF)(CB)2"] += moved
    _add_corrections(frag, canonical_for_corrections)
    frag.symmetry_number = _symmetry_number(mol)
    frag.optical_isomers = _optical_isomer_count(mol)
    return frag


def _add_corrections(
    frag: Fragmentation,
    canonical_for_corrections: str | None = None,
) -> None:
    mol = frag.mol
    corr = frag.corrections
    canonical = (canonical_for_corrections
                 if canonical_for_corrections is not None
                 else Chem.MolToSmiles(mol, isomericSmiles=True))
    named_correction = _NAMED_STRUCTURE_CORRECTIONS.get(canonical)
    if named_correction is not None:
        corr[named_correction] += 1

    # Methyl repulsion/branching corrections (Table 3).  A methyl bonded to a
    # tertiary or quaternary sp3 carbon contributes one correction.
    tertiary = set()
    quaternary = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 7) or atom.GetIsAromatic() or atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        carbon_neighbors = sum(nb.GetAtomicNum() == 6 for nb in atom.GetNeighbors())
        if (atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 1
                and atom.GetDegree() == 3 and carbon_neighbors >= 2):
            tertiary.add(atom.GetIdx())
        elif ((atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 0
               and atom.GetDegree() == 4 and carbon_neighbors >= 3)
              or (atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() == 0 and atom.GetDegree() == 3)):
            quaternary.add(atom.GetIdx())
    methyl_hosts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() != 3:
            continue
        host = next(iter(atom.GetNeighbors()), None)
        if host is None:
            continue
        if host.GetIdx() in tertiary or host.GetIdx() in quaternary:
            methyl_hosts.append(host.GetIdx())
    carbon_quaternary = {
        idx for idx in quaternary
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    }
    if len(carbon_quaternary) >= 2:
        corr["CH3-quat-quat"] += len(methyl_hosts)
    elif tertiary and carbon_quaternary:
        corr["CH3-tert-quat"] += len(methyl_hosts)
    else:
        n_tertiary = sum(idx in tertiary for idx in methyl_hosts)
        n_quaternary = sum(idx in quaternary for idx in methyl_hosts)
        if n_tertiary:
            corr["CH3-tertiary"] += n_tertiary
        if n_quaternary:
            corr["CH3-quaternary"] += n_quaternary

    acid_carbons = _carboxylic_acid_carbons(mol)
    amino_nitrogens = _primary_amine_nitrogens(mol)
    if acid_carbons and amino_nitrogens:
        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if not aromatic:
            corr["zwitterion-aliphatic"] += 1
        else:
            directly_conjugated = any(
                any(nb.GetIsAromatic() for nb in atom.GetNeighbors())
                for atom in (*acid_carbons, *amino_nitrogens))
            corr["zwitterion-aromatic-II" if directly_conjugated
                 else "zwitterion-aromatic-I"] += 1

    rings = mol.GetRingInfo().AtomRings()
    if len(rings) == 1 and named_correction is None:
        ring = rings[0]
        ring_set = set(ring)
        substituted = any(
            any(nb.GetIdx() not in ring_set for nb in mol.GetAtomWithIdx(i).GetNeighbors())
            for i in ring)
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if not aromatic:
            double_cc = sum(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
                for bond in mol.GetBonds())
            hetero = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
            carbonyl_in_ring = any(_is_carbonyl(mol.GetAtomWithIdx(i)) for i in ring)
            key = None
            if carbonyl_in_ring and not hetero and len(ring) in (5, 6, 7, 8, 9, 10, 11, 12, 15, 17):
                key = {
                    5: "cyclopentanone-rsc", 6: "cyclohexanone-rsc",
                    7: "cycloheptanone-rsc", 8: "cyclooctanone-rsc",
                    9: "cyclononanone-rsc", 10: "cyclodecanone-rsc",
                    11: "cycloundecanone-rsc", 12: "cyclododecanone-rsc",
                    15: "cyclopentadecanone-rsc", 17: "cycloheptadecanone-rsc",
                }[len(ring)]
            elif hetero == [8] and double_cc == 0 and len(ring) in (3, 4, 5, 6):
                key = {3: "ethylene-oxide-rsc", 4: "trimethylene-oxide-rsc", 5: "tetrahydrofuran-rsc", 6: "tetrahydropyran-rsc"}[len(ring)]
            elif hetero == [16] and double_cc == 0 and len(ring) in (3, 4, 5, 6, 7):
                key = {3: "thiacyclopropane-rsc", 4: "thiacyclobutane-rsc", 5: "thiacyclopentane-rsc", 6: "thiacyclohexane-rsc", 7: "thiacycloheptane-rsc"}[len(ring)]
            elif not hetero and double_cc == 0 and len(ring) in range(3, 18):
                if len(ring) in (3, 5, 6):
                    key = f"cyclo{ {3:'propane', 5:'pentane', 6:'hexane'}[len(ring)]}-{'sub' if substituted else 'unsub'}-rsc"
                else:
                    key = {
                        4: "cyclobutane-rsc", 7: "cycloheptane-rsc",
                        8: "cyclooctane-rsc", 9: "cyclononane-rsc",
                        10: "cyclodecane-rsc", 11: "cycloundecane-rsc",
                        12: "cyclododecane-rsc", 13: "cyclotridecane-rsc",
                        14: "cyclotetradecane-rsc", 15: "cyclopentadecane-rsc",
                        16: "cyclohexadecane-rsc", 17: "cycloheptadecane-rsc",
                    }[len(ring)]
            elif not hetero and double_cc == 1 and len(ring) in range(3, 9):
                if len(ring) == 5:
                    key = "cyclopentene-sub-rsc" if substituted else "cyclopentene-unsub-rsc"
                else:
                    key = {3: "cyclopropene-rsc", 4: "cyclobutene-rsc", 6: "cyclohexene-rsc", 7: "cycloheptene-rsc", 8: "cyclooctene-rsc"}[len(ring)]
            if key is not None:
                corr[key] += 1

    # Naphthalene has a published zero-H correction but nonzero Cp/S terms.
    aromatic_rings = [set(r) for r in rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if len(aromatic_rings) == 2 and len(aromatic_rings[0] | aromatic_rings[1]) == 10 and len(aromatic_rings[0] & aromatic_rings[1]) == 2:
        core = aromatic_rings[0] | aromatic_rings[1]
        substitutions = sum(
            any(nb.GetIdx() not in core for nb in mol.GetAtomWithIdx(i).GetNeighbors())
            for i in core)
        corr[f"naphthalene-{min(substitutions, 2)}-sub" if substitutions else "naphthalene-unsub"] += 2

    # Ortho/meta hydrocarbon corrections on isolated six-member aromatic rings.
    for ring in rings:
        if len(ring) != 6 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(sum(bond.GetIsAromatic() for bond in mol.GetAtomWithIdx(i).GetBonds()) == 3
               for i in ring):
            continue
        substituted_positions = []
        for pos, idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(idx)
            exocyclic = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in ring]
            if exocyclic:
                substituted_positions.append(
                    (pos, _aromatic_substituent_label(atom, exocyclic[0])))
        for n, (left, label_left) in enumerate(substituted_positions):
            for right, label_right in substituted_positions[n + 1:]:
                distance = min((right - left) % 6, (left - right) % 6)
                key = _AROMATIC_PAIR_CORRECTIONS.get(
                    (distance, frozenset((label_left, label_right))))
                if key is None and distance == 1 and (
                        label_left in {"alkyl", "CH3"}
                        and label_right in {"F", "Cl", "Br", "I"}
                        or label_right in {"alkyl", "CH3"}
                        and label_left in {"F", "Cl", "Br", "I"}):
                    key = "ortho-alkyl-X"
                if key is not None:
                    corr[key] += 1

    # 2,2'-halogen interactions across a biphenyl single bond use the primed
    # corrections in Table 2, distinct from two substituents on one ring.
    aromatic_rings_by_atom = {
        idx: [set(ring) for ring in rings if idx in ring
              and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
        for idx in range(mol.GetNumAtoms())
    }
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if (bond.GetIsAromatic() or not left.GetIsAromatic()
                or not right.GetIsAromatic()):
            continue
        labels = []
        for endpoint, partner in ((left, right), (right, left)):
            endpoint_labels = []
            for ring in aromatic_rings_by_atom[endpoint.GetIdx()]:
                for neighbor in endpoint.GetNeighbors():
                    if neighbor.GetIdx() not in ring or neighbor.GetIdx() == partner.GetIdx():
                        continue
                    for substituent in neighbor.GetNeighbors():
                        if substituent.GetIdx() not in ring:
                            endpoint_labels.append(
                                _aromatic_substituent_label(neighbor, substituent))
            labels.append(endpoint_labels)
        for halogen, correction in (("F", "ortho-F-F-prime"),
                                     ("Cl", "ortho-Cl-Cl-prime")):
            if halogen in labels[0] and halogen in labels[1]:
                corr[correction] += 1

    # Adjacent aliphatic functional groups (the aromatic versions are handled
    # by the ring-position table above).
    nitro_hosts = set()
    nitrate_hosts = set()
    for atom in mol.GetAtoms():
        nitro = _nitro_atoms(atom)
        if nitro:
            host = next((nb for nb in atom.GetNeighbors()
                         if nb.GetAtomicNum() != 8), None)
            if host is not None and not host.GetIsAromatic():
                nitro_hosts.add(host.GetIdx())
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 2:
            neighbors = list(atom.GetNeighbors())
            carbon = next((nb for nb in neighbors if nb.GetAtomicNum() == 6), None)
            nitrogen = next((nb for nb in neighbors if nb.GetAtomicNum() == 7), None)
            if (carbon is not None and nitrogen is not None
                    and len([nb for nb in nitrogen.GetNeighbors()
                             if nb.GetAtomicNum() == 8]) == 3):
                nitrate_hosts.add(carbon.GetIdx())
    for bond in mol.GetBonds():
        pair = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        if pair <= nitro_hosts:
            corr["NO2-NO2-aliphatic-adjacent"] += 1
        if pair <= nitrate_hosts:
            corr["ONO2-ONO2-aliphatic-adjacent"] += 1

    # Pyridine-like nearest-neighbour corrections.
    for atom in mol.GetAtoms():
        if not (atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                and atom.GetTotalNumHs() == 0):
            continue
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 7 and nb.GetIsAromatic():
                if atom.GetIdx() < nb.GetIdx():
                    corr["Nr-Nr-ortho"] += 1
                continue
            if not (nb.GetAtomicNum() == 6 and nb.GetIsAromatic()):
                continue
            if any(ext.GetAtomicNum() == 6 and ext.GetTotalNumHs() == 3
                   for ext in nb.GetNeighbors() if ext.GetIdx() != atom.GetIdx()):
                corr["Nr-CH3-ortho"] += 1

    def alkene_label(atom: Chem.Atom, partner: Chem.Atom) -> str:
        others = [nb for nb in atom.GetNeighbors()
                  if nb.GetIdx() != partner.GetIdx()]
        if not others:
            return "H"
        other = others[0]
        if other.GetAtomicNum() in (9, 17, 35, 53):
            return other.GetSymbol()
        if other.GetAtomicNum() == 16 and _sulfur_oxide_count(other) == 1:
            return "SO"
        if other.GetAtomicNum() == 6:
            if other.GetTotalNumHs() == 3:
                return "CH3"
            if any(b.GetBondType() == Chem.BondType.TRIPLE
                   and b.GetOtherAtom(other).GetAtomicNum() == 7
                   for b in other.GetBonds()):
                return "CN"
            if (other.GetHybridization() == Chem.HybridizationType.SP3
                    and other.GetTotalNumHs() == 0
                    and sum(nb.GetAtomicNum() == 6
                            for nb in other.GetNeighbors()) == 4):
                return "tert-butyl"
        return "other"

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() != Chem.BondStereo.STEREOZ:
            continue
        if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6:
            left, right = bond.GetBeginAtom(), bond.GetEndAtom()
            pair = frozenset((alkene_label(left, right),
                              alkene_label(right, left)))
            special = {
                frozenset(("Cl", "Cl")): "cis-Cl-Cl",
                frozenset(("I", "I")): "cis-I-I",
                frozenset(("CH3", "Br")): "cis-CH3-Br",
                frozenset(("CH3", "I")): "cis-CH3-I",
                frozenset(("CH3", "CN")): "CH3-CN-cis",
            }.get(pair)
            if "SO" in pair:
                corr["cis-sulfoxide"] += 1
            elif "tert-butyl" in pair:
                corr["tert-butyl-cis"] += 1
            elif special is not None:
                corr[special] += 1
            else:
                corr["cis-unsaturation"] += 1


def _symmetry_number(mol: Chem.Mol) -> int:
    """Graph-derived total symmetry number used by the paper's gas entropy.

    Heavy-atom automorphisms supply external symmetry, terminal methyl groups
    supply their threefold internal rotations, and tetrahedral reflection
    automorphisms are removed.  This reproduces the paper's tabulated sigma
    for methane, linear/branched alkanes, benzene, toluene, and alcohol tests.
    """
    if mol.GetNumHeavyAtoms() == 1 and mol.GetAtomWithIdx(0).GetAtomicNum() == 6 and mol.GetAtomWithIdx(0).GetTotalNumHs() == 4:
        return 12
    if (mol.GetNumHeavyAtoms() == 2
            and all(atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2
                    and atom.GetHybridization() == Chem.HybridizationType.SP2
                    for atom in mol.GetAtoms())):
        return 4
    automorphisms = mol.GetSubstructMatches(
        mol, uniquify=False, useChirality=True, maxMatches=100_000)
    external = max(1, len(automorphisms))
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) == 1:
        ring = rings[0]
        ring_set = set(ring)
        if (len(ring) in (3, 4, 5, 6, 7, 8)
                and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)
                and all(not bond.GetIsAromatic()
                        and bond.GetBondType() == Chem.BondType.SINGLE
                        for bond in mol.GetBonds()
                        if bond.GetBeginAtomIdx() in ring_set
                        and bond.GetEndAtomIdx() in ring_set)
                and all(all(nb.GetIdx() in ring_set
                            for nb in mol.GetAtomWithIdx(i).GetNeighbors())
                        for i in ring)):
            # Total symmetry numbers printed for the equilibrium ring
            # conformers in Tables 12/13; graph dihedral order alone is wrong
            # for puckered cyclohexane/cycloheptane.
            return {3: 6, 4: 8, 5: 10, 6: 6, 7: 2, 8: 8}[len(ring)]
    rigid_orientation_filter = len(rings) > 1 and any(
        atom.GetHybridization() == Chem.HybridizationType.SP3
        and atom.GetDegree() >= 3 for atom in mol.GetAtoms())
    if rigid_orientation_filter:
        external = _proper_rigid_automorphism_count(mol, automorphisms)
    else:
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in (6, 7) or atom.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            carbon_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6]
            if len(carbon_neighbors) < 2 or atom.GetTotalNumHs() > 1:
                continue
            terminal_methyl_ranks = [
                ranks[nb.GetIdx()] for nb in carbon_neighbors
                if nb.GetTotalNumHs() == 3 and nb.GetDegree() == 1]
            all_carbon_center = (atom.GetAtomicNum() == 6
                                 and all(nb.GetAtomicNum() == 6 for nb in atom.GetNeighbors()))
            if ((all_carbon_center or atom.GetAtomicNum() == 7)
                    and len(terminal_methyl_ranks) >= 2
                    and len(set(terminal_methyl_ranks)) < len(terminal_methyl_ranks)
                    and external % 2 == 0):
                external //= 2
        for ring in rings:
            if len(ring) != 6 or not all(
                    mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            substituted = []
            for position, idx in enumerate(ring):
                outside = [nb for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if nb.GetIdx() not in ring]
                if outside:
                    substituted.append((position, outside[0]))
            if len(substituted) != 2:
                continue
            (left_pos, left), (right_pos, right) = substituted
            distance = min((right_pos - left_pos) % 6, (left_pos - right_pos) % 6)
            if (distance == 3 and ranks[left.GetIdx()] == ranks[right.GetIdx()]
                    and any(atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 3
                            for atom in mol.GetAtoms())
                    and external % 2 == 0):
                external //= 2
    methyl_rotors = sum(
        atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 3 and atom.GetDegree() == 1
        for atom in mol.GetAtoms())
    return max(1, external * 3 ** methyl_rotors)


def _tetrahedral_orientation(
    conformer: Chem.Conformer,
    center: int,
    neighbors: list[int],
) -> float:
    """Signed volume for one ordered tetrahedral neighbour set."""
    points = [conformer.GetAtomPosition(index) for index in neighbors]
    a = points[0] - points[3]
    b = points[1] - points[3]
    c = points[2] - points[3]
    return (
        a.x * (b.y * c.z - b.z * c.y)
        - a.y * (b.x * c.z - b.z * c.x)
        + a.z * (b.x * c.y - b.y * c.x)
    )


def _proper_rigid_automorphism_count(
    mol: Chem.Mol,
    automorphisms: tuple[tuple[int, ...], ...],
) -> int:
    """Exclude reflection automorphisms for rigid nonplanar skeletons.

    A heavy-atom graph cannot distinguish rotations from reflections.  For a
    rigid polycycle, tetrahedral orientations from one deterministic 3-D
    embedding provide that missing distinction.  Centers with two or more
    hydrogens impose no orientation constraint because exchanging equivalent
    hydrogens can reverse their local ordering.
    """
    with_hydrogens = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xD04A15
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(with_hydrogens, params) != 0:
        raise FragmentationError(
            "could not determine the rotational symmetry of a rigid polycycle")
    conformer = with_hydrogens.GetConformer()
    centers: list[tuple[int, list[int]]] = []
    heavy_count = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        full_atom = with_hydrogens.GetAtomWithIdx(atom.GetIdx())
        neighbors = [neighbor.GetIdx() for neighbor in full_atom.GetNeighbors()]
        hydrogen_count = sum(index >= heavy_count for index in neighbors)
        if len(neighbors) == 4 and hydrogen_count <= 1:
            centers.append((atom.GetIdx(), neighbors))
    if not centers:
        return max(1, len(automorphisms))

    proper = 0
    for mapping in automorphisms:
        preserves_orientation = True
        for center, neighbors in centers:
            target_center = mapping[center]
            mapped_neighbors = []
            for neighbor in neighbors:
                if neighbor < heavy_count:
                    mapped_neighbors.append(mapping[neighbor])
                else:
                    target_hydrogens = [
                        item.GetIdx() for item in with_hydrogens.GetAtomWithIdx(
                            target_center).GetNeighbors()
                        if item.GetAtomicNum() == 1]
                    if len(target_hydrogens) != 1:
                        raise FragmentationError(
                            "inconsistent tetrahedral hydrogen mapping while "
                            "determining rotational symmetry")
                    mapped_neighbors.append(target_hydrogens[0])
            source_sign = _tetrahedral_orientation(
                conformer, center, neighbors)
            target_sign = _tetrahedral_orientation(
                conformer, target_center, mapped_neighbors)
            if source_sign * target_sign < 0.0:
                preserves_orientation = False
                break
        if preserves_orientation:
            proper += 1
    return max(1, proper)


def _stereoisomer_optical_multiplicity(mol: Chem.Mol) -> int:
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mirror = Chem.Mol(mol)
    for atom in mirror.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(mirror, cleanIt=False, force=True)
    mirror_canonical = Chem.MolToSmiles(mirror, isomericSmiles=True)
    return 1 if canonical == mirror_canonical else 2


def _optical_isomer_count(mol: Chem.Mol) -> int | None:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=False)
    if not centers:
        return 1
    unassigned = [index for index, label in centers if label == "?"]
    if not unassigned:
        return _stereoisomer_optical_multiplicity(mol)
    if len(unassigned) > 12:
        return None

    multiplicities = set()
    for bits in itertools.product((False, True), repeat=len(unassigned)):
        isomer = Chem.Mol(mol)
        for index, clockwise in zip(unassigned, bits):
            isomer.GetAtomWithIdx(index).SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CW if clockwise
                else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        Chem.AssignStereochemistry(isomer, cleanIt=True, force=True)
        multiplicities.add(_stereoisomer_optical_multiplicity(isomer))
    if len(multiplicities) != 1:
        return None
    return multiplicities.pop()


def _element_entropy_sum(mol: Chem.Mol) -> float:
    counts = Counter(atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms())
    total = 0.0
    for atomic_number, count in counts.items():
        coefficient = count / 2.0 if atomic_number in _DIATOMIC_ELEMENTS else float(count)
        total += coefficient * ELEMENT_ENTROPIES[atomic_number]
    return total


def _sum_property(frag: Fragmentation, phase: str, cell: int) -> float | None:
    total = 0.0
    for name, count in frag.groups.items():
        value = GROUP_VALUES[name].phase(phase)[cell]
        if value is None:
            return None
        total += count * value
    for name, count in frag.corrections.items():
        value = CORRECTION_VALUES[name].phase(phase)[cell]
        if value is None:
            return None
        total += count * value
    if phase == "gas" and cell == 2:
        if frag.optical_isomers is None:
            return None
        total -= R_J_MOL_K * math.log(frag.symmetry_number)
        total += R_J_MOL_K * math.log(frag.optical_isomers)
    return total


def _phase_result(frag: Fragmentation, phase: str, element_entropy: float) -> PhaseResult:
    enthalpy = _sum_property(frag, phase, 0)
    cp = _sum_property(frag, phase, 1)
    entropy = _sum_property(frag, phase, 2)
    entropy_formation = None if entropy is None else entropy - element_entropy
    gibbs = None
    ln_k = None
    if enthalpy is not None and entropy_formation is not None:
        gibbs = enthalpy - TEMPERATURE_K * entropy_formation / 1000.0
        ln_k = -gibbs * 1000.0 / (R_J_MOL_K * TEMPERATURE_K)
    return PhaseResult(enthalpy, cp, entropy, entropy_formation, gibbs, ln_k)


def _parse_smiles(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise DomalskiHearingError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DomalskiHearingError(f"could not parse SMILES {smiles!r}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise FragmentationError("the method requires one molecular fragment")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _ALLOWED_ATOMIC_NUMBERS:
            raise FragmentationError(
                f"element {atom.GetSymbol()} is outside the published "
                "C/H/N/O/S/halogen domain")
        if atom.GetNumRadicalElectrons():
            raise FragmentationError(
                "radicals are outside the published molecular domain")
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        raise FragmentationError(
            "structures with a nonzero net charge are outside the published "
            "molecular domain")
    return mol


@lru_cache(maxsize=4096)
def estimate(smiles: str) -> DomalskiHearingResult:
    """Estimate all published Domalski--Hearing values from a SMILES string.

    The structure must be one neutral, closed-shell molecule containing only
    C, H, N, O, S, F, Cl, Br, and I.  A :class:`FragmentationError` identifies
    a structural environment for which the 1993 table publishes no group.
    """
    mol = _parse_smiles(smiles)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    frag = _assign_groups(canonical, mol)
    element_entropy = _element_entropy_sum(mol)
    warnings = []
    if frag.optical_isomers is None:
        warnings.append(
            "gas entropy is unavailable because unspecified stereocenters "
            "admit different optical-isomer multiplicities; specify stereochemistry")
    for phase in ("gas", "liquid", "solid"):
        missing = []
        for name in (*frag.groups.keys(), *frag.corrections.keys()):
            row = (GROUP_VALUES.get(name) or CORRECTION_VALUES[name]).phase(phase)
            if any(value is None for value in row):
                missing.append(name)
        if missing:
            warnings.append(
                f"{phase}: one or more properties are unavailable where Table 2 is blank "
                f"({', '.join(sorted(set(missing)))})")
    return DomalskiHearingResult(
        smiles=canonical,
        temperature_K=TEMPERATURE_K,
        pressure_Pa=PRESSURE_PA,
        gas=_phase_result(frag, "gas", element_entropy),
        liquid=_phase_result(frag, "liquid", element_entropy),
        solid=_phase_result(frag, "solid", element_entropy),
        groups=dict(sorted(frag.groups.items())),
        corrections=dict(sorted(frag.corrections.items())),
        symmetry_number=frag.symmetry_number,
        optical_isomers=frag.optical_isomers,
        warnings=tuple(warnings),
    )


def fragment(smiles: str) -> Fragmentation:
    """Return the strict published-group fragmentation for inspection."""
    mol = _parse_smiles(smiles)
    return _assign_groups(Chem.MolToSmiles(mol, isomericSmiles=True), mol)


_HF_NITRILE_OXIDE = Chem.MolFromSmarts("[C]#[N+][O-]")


def has_sulfur_sulfur_bond(result: DomalskiHearingResult) -> bool:
    """Return whether the resolved molecular structure contains an S-S bond."""
    molecule = Chem.MolFromSmiles(result.smiles)
    if molecule is None:
        return False
    return any(
        bond.GetBeginAtom().GetAtomicNum()
        == bond.GetEndAtom().GetAtomicNum()
        == 16
        for bond in molecule.GetBonds()
    )


def assess_hf_applicability(result: DomalskiHearingResult) -> HfApplicability:
    """Assess the validated structural applicability domain of gas Hf.

    The estimator can produce a formal group sum for several structural
    families whose experimental Hf errors are systematically unreliable.
    This assessment is deliberately property-specific: it does not restrict
    the independently validated entropy or heat-capacity estimates.
    """
    molecule = Chem.MolFromSmiles(result.smiles)
    if molecule is None:
        return HfApplicability(False, ("SMILES could not be parsed",))

    rings = molecule.GetRingInfo().AtomRings()
    ring_memberships = [
        sum(atom_index in ring for ring in rings)
        for atom_index in range(molecule.GetNumAtoms())
    ]
    ring_strain_correction = any(
        "rsc" in name.casefold() for name in result.corrections
    )
    has_small_ring = any(len(ring) <= 4 for ring in rings)
    fused_pairs = [
        (first, second)
        for index, first in enumerate(rings)
        for second in rings[index + 1:]
        if len(set(first) & set(second)) >= 2
    ]

    reasons = []
    if has_small_ring and (len(rings) > 1 or not ring_strain_correction):
        reasons.append("uncorrected or polycyclic three-/four-membered ring")
    if (
        rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
        + rdMolDescriptors.CalcNumSpiroAtoms(molecule)
    ):
        reasons.append("bridged or spiro ring topology")
    if max(ring_memberships, default=0) >= 3:
        reasons.append("atom shared by at least three rings")
    if len(rings) >= 4:
        reasons.append("four or more rings")
    if any(
        len(first) not in (5, 6) or len(second) not in (5, 6)
        for first, second in fused_pairs
    ):
        reasons.append("unusual fused-ring sizes")
    if has_sulfur_sulfur_bond(result):
        reasons.append("sulfur-sulfur bond")
    if any(
        bond.GetBeginAtom().GetAtomicNum()
        == bond.GetEndAtom().GetAtomicNum()
        == 8
        for bond in molecule.GetBonds()
    ):
        reasons.append("oxygen-oxygen bond")
    if (
        _HF_NITRILE_OXIDE is not None
        and molecule.HasSubstructMatch(_HF_NITRILE_OXIDE)
    ):
        reasons.append("nitrile oxide")
    if sum(result.corrections.values()) >= 6:
        reasons.append("at least six accumulated structural corrections")
    return HfApplicability(not reasons, tuple(reasons))


__all__ = [
    "AMBIGUOUS_PAPER_CORRECTION_VALUES",
    "CORRECTION_VALUES",
    "DomalskiHearingError",
    "DomalskiHearingResult",
    "Fragmentation",
    "FragmentationError",
    "GROUP_VALUES",
    "HfApplicability",
    "PhaseResult",
    "assess_hf_applicability",
    "estimate",
    "fragment",
    "has_sulfur_sulfur_bond",
]
