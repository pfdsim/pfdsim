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
