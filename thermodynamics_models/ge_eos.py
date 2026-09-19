"""Small shared abstractions for excess-Gibbs-energy cubic EOS mixing rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ExcessGibbsState:
    """Dimensionless excess-Gibbs model values at fixed composition."""

    ln_activity_coefficients: tuple[float, ...]
    dln_activity_coefficients_dT: tuple[float, ...]

    def g_over_rt(self, composition: Sequence[float]) -> float:
        return sum(
            mole_fraction * value
            for mole_fraction, value in zip(
                composition, self.ln_activity_coefficients
            )
        )

    def d_g_over_rt_dT(self, composition: Sequence[float]) -> float:
        return sum(
            mole_fraction * value
            for mole_fraction, value in zip(
                composition, self.dln_activity_coefficients_dT
            )
        )


class ExcessGibbsModel(Protocol):
    """Interface expected by a future family of gE-EOS mixing rules."""

    def excess_gibbs_state(
        self,
        composition: tuple[float, ...],
        T_K: float,
    ) -> ExcessGibbsState:
        ...


@dataclass(frozen=True)
class GEOSMixingState:
    """Common dimensionless cubic-EOS mixing outputs."""

    b: float
    D: float
    dD_dT: float
    composition_derivatives: tuple[float, ...]


class ExcessGibbsEOSMixingRule(Protocol):
    def mix(
        self,
        composition: tuple[float, ...],
        pure_b: tuple[float, ...],
        pure_D: tuple[float, ...],
        pure_dD_dT: tuple[float, ...],
        excess: ExcessGibbsState,
    ) -> GEOSMixingState:
        ...


@dataclass(frozen=True)
class ModifiedHuronVidalFirstOrderMixingRule:
    """First-order modified Huron-Vidal mixing used by published PSRK."""

    q1: float

    def mix(
        self,
        composition: tuple[float, ...],
        pure_b: tuple[float, ...],
        pure_D: tuple[float, ...],
        pure_dD_dT: tuple[float, ...],
        excess: ExcessGibbsState,
    ) -> GEOSMixingState:
        if not composition or not (
            len(composition)
            == len(pure_b)
            == len(pure_D)
            == len(pure_dD_dT)
            == len(excess.ln_activity_coefficients)
            == len(excess.dln_activity_coefficients_dT)
        ):
            raise ValueError("gE-EOS mixing inputs must have one value per component")
        if not math.isfinite(self.q1) or self.q1 == 0.0:
            raise ValueError("gE-EOS mixing q1 must be finite and nonzero")

        b_mix = sum(
            mole_fraction * b_value
            for mole_fraction, b_value in zip(composition, pure_b)
        )
        if not math.isfinite(b_mix) or b_mix <= 0.0:
            raise ValueError("gE-EOS mixture b must be positive and finite")
        size_term = sum(
            mole_fraction * math.log(b_mix / b_value)
            for mole_fraction, b_value in zip(composition, pure_b)
        )
        D = (
            sum(
                mole_fraction * value
                for mole_fraction, value in zip(composition, pure_D)
            )
            + (excess.g_over_rt(composition) + size_term) / self.q1
        )
        dD_dT = (
            sum(
                mole_fraction * value
                for mole_fraction, value in zip(composition, pure_dD_dT)
            )
            + excess.d_g_over_rt_dT(composition) / self.q1
        )
        derivatives = tuple(
            pure_value
            + (
                ln_gamma
                + math.log(b_mix / b_value)
                + b_value / b_mix
                - 1.0
            )
            / self.q1
            for pure_value, ln_gamma, b_value in zip(
                pure_D, excess.ln_activity_coefficients, pure_b
            )
        )
        # gE-EOS mixing rules can legitimately produce a negative mixture
        # attraction parameter for strongly asymmetric supercritical gas
        # mixtures. The cubic state remains admissible as long as D and its
        # derivatives are finite and a physical root exists.
        if not math.isfinite(D):
            raise ValueError("gE-EOS mixture D must be finite")
        if not math.isfinite(dD_dT) or any(
            not math.isfinite(value) for value in derivatives
        ):
            raise ValueError("gE-EOS mixing derivatives must be finite")
        return GEOSMixingState(
            b=b_mix,
            D=D,
            dD_dT=dD_dT,
            composition_derivatives=derivatives,
        )


@dataclass(frozen=True)
class ModifiedHuronVidalSecondOrderMixingRule:
    """Dahl–Michelsen MHV2 for SRK; D = a/(bRT).

    Composition derivatives are partial molar nD, not unconstrained dD/dx.
    They enter fugacity directly and their mole-fraction average equals D.
    """

    q1: float = -0.478
    q2: float = -0.0047

    def mix(self, composition, pure_b, pure_D, pure_dD_dT, excess):
        lengths = {len(composition), len(pure_b), len(pure_D), len(pure_dD_dT),
                   len(excess.ln_activity_coefficients),
                   len(excess.dln_activity_coefficients_dT)}
        if len(lengths) != 1 or not composition:
            raise ValueError('MHV2 mixing inputs must have one value per component')
        if not (math.isfinite(self.q1) and self.q1 < 0.0
                and math.isfinite(self.q2) and self.q2 < 0.0):
            raise ValueError('MHV2 q1 and q2 must be negative and finite')
        if any(not math.isfinite(v) or v <= 0 for v in pure_b):
            raise ValueError('MHV2 pure covolumes must be positive and finite')
        b = sum(x * v for x, v in zip(composition, pure_b))
        pure_q = tuple(self.q1 * d + self.q2 * d * d for d in pure_D)
        size = tuple(math.log(b / bi) for bi in pure_b)
        rhs = excess.g_over_rt(composition) + sum(
            x * (qi + si) for x, qi, si in zip(composition, pure_q, size))
        discriminant = self.q1**2 + 4.0 * self.q2 * rhs
        if not math.isfinite(discriminant) or discriminant <= 0.0:
            raise ValueError('MHV2 mixing equation has no nonsingular physical branch')
        # Stable quadratic root continuous with pure-component attraction.
        D = 2.0 * rhs / (self.q1 - math.sqrt(discriminant))
        slope = self.q1 + 2.0 * self.q2 * D
        dD_dT = (excess.d_g_over_rt_dT(composition) + sum(
            x * (self.q1 + 2.0 * self.q2 * d) * dd
            for x, d, dd in zip(composition, pure_D, pure_dD_dT))) / slope
        sigma = tuple((qi + self.q2 * D * D + lg + si + bi / b - 1.0) / slope
                      for qi, lg, si, bi in zip(pure_q,
                          excess.ln_activity_coefficients, size, pure_b))
        if any(not math.isfinite(v) for v in (D, dD_dT, *sigma)):
            raise ValueError('MHV2 mixing state must be finite')
        return GEOSMixingState(b, D, dD_dT, sigma)
