"""
Vapor Dimerization Model (VDM)

Corrects vapor-phase fugacity coefficients for association reactions.
Supports single-species dimerization, 2A ⇌ A₂, and mixed acid vapors with
cross-association, A + B ⇌ AB.

Fugacity coefficient decomposition:
    ln φ_i = ln φ_i^P + ln φ_i^D

where:
  - φ_i^P  = physical fugacity coefficient (from RK EOS at the true,
             post-association composition)
  - φ_i^D  = chemical fugacity coefficient = y_i / y_i° , the ratio of
             true to nominal (stoichiometric) mole fractions after solving
             the dimerization equilibrium.

Equilibrium:  K(T) = (a_A₂) / (a_A)²
with fugacity-based activities referenced to 1 bar standard state:
    K = (φ_A₂^P · y_A₂ · P / 1 bar) / (φ_A^P · y_A · P / 1 bar)²

Temperature dependence via van't Hoff:
    ln( K(T) / K_ref ) = -(ΔH / R) · (1/T - 1/T_ref)
"""

import math
import warnings
from functools import lru_cache
from typing import Optional

from pathlib import Path

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

R = R_J_MOL_K          # J/mol·K
P_STD = 1.0        # bar (standard state for fugacity-based K)
DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def _compiled_vdm_backend():
    """Load the optional compiled backend once without eager global imports."""
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from . import compiled_vdm
    else:
        import compiled_vdm
    return compiled_vdm


GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL = -60500.0
GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K = -144.0
CROSS_DIMER_STATISTICAL_DELTA_S_J_MOL_K = R * math.log(2.0)


def _normalize_nominal_composition_for_dimers(
    y_nominal: dict[str, float],
    dimer_symbols: set[str],
) -> dict[str, float]:
    """Return nominal monomer/inert composition with internal dimers removed."""
    for dimer in dimer_symbols:
        if dimer in y_nominal and abs(y_nominal[dimer]) > 1e-14:
            raise ValueError(
                f"Nominal VDM composition should not include internal dimer "
                f"species '{dimer}'."
            )

    cleaned = {}
    for comp, yi in y_nominal.items():
        if comp in dimer_symbols:
            continue
        if yi < -1e-14:
            raise ValueError(f"Negative nominal mole fraction for '{comp}': {yi}")
        cleaned[comp] = max(float(yi), 0.0)

    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("Nominal VDM composition must contain a positive total mole fraction.")
    return {comp: yi / total for comp, yi in cleaned.items()}


class VaporDimerizationModel:
    """
    Vapor-phase dimerization fugacity correction.

    Parameters
    ----------
    monomer : str
        Symbol of the dimerizing species (e.g. 'CH3COOH').
    dimer : str
        Symbol of the dimer species (e.g. '(CH3COOH)2').
    delta_H : float
        Enthalpy of dimerization [J/mol]. Negative for exothermic
        association (typical for hydrogen-bonded dimers).

    Specify K via ONE of:
        K_ref, T_ref : Equilibrium constant at reference temperature,
            using the 1 bar fugacity standard.
        delta_S : Entropy of dimerization [J/mol·K].  Then
            K(T) = exp(ΔS/R - ΔH/(RT)).
    """

    def __init__(
        self,
        monomer: str,
        dimer: str,
        delta_H: float = -60000.0,
        K_ref: Optional[float] = None,
        T_ref: float = 298.15,
        delta_S: Optional[float] = None,
    ):
        if K_ref is None and delta_S is None:
            raise ValueError("Provide either K_ref or delta_S (or both).")

        self.monomer = monomer
        self.dimer = dimer
        self.delta_H = delta_H

        if delta_S is not None:
            # Thermodynamic formulation:  K(T) = exp(ΔS/R - ΔH/(RT))
            self._delta_S = delta_S
            self._use_delta_S = True
            # Store K_ref and T_ref for compatibility (compute at T_ref)
            self.T_ref = T_ref
            self.K_ref = math.exp(
                delta_S / R - delta_H / (R * T_ref)
            )
        else:
            self._delta_S = None
            self._use_delta_S = False
            self.K_ref = float(K_ref)  # type: ignore[arg-type]
            self.T_ref = T_ref
        self._association_state_cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Equilibrium constant
    # ------------------------------------------------------------------

    def K_eq(self, T: float) -> float:
        """Equilibrium constant K(T), using a 1 bar fugacity standard."""
        if self._use_delta_S:
            return math.exp(
                (self._delta_S / R) - (self.delta_H / (R * T))
            )
        return self.K_ref * math.exp(
            -(self.delta_H / R) * (1.0 / T - 1.0 / self.T_ref)
        )

    def association_enthalpy(self) -> float:
        """Return ΔH for dimerization [J/mol]."""
        return self.delta_H

    def association_entropy(self) -> float:
        """Return ΔS for dimerization [J/mol/K]."""
        if self._delta_S is not None:
            return self._delta_S
        return R * math.log(self.K_ref) + self.delta_H / self.T_ref

    def _normalize_nominal_composition(self, y_nominal: dict[str, float]) -> dict[str, float]:
        """Return a clean nominal composition without the internal dimer species."""
        if self.dimer in y_nominal and abs(y_nominal[self.dimer]) > 1e-14:
            raise ValueError(
                f"Nominal VDM composition should not include internal dimer "
                f"species '{self.dimer}'."
            )

        cleaned = {}
        for comp, yi in y_nominal.items():
            if comp == self.dimer:
                continue
            if yi < -1e-14:
                raise ValueError(f"Negative nominal mole fraction for '{comp}': {yi}")
            cleaned[comp] = max(float(yi), 0.0)

        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("Nominal VDM composition must contain a positive total mole fraction.")
        return {comp: yi / total for comp, yi in cleaned.items()}

    @staticmethod
    def _alpha_from_equilibrium(kappa: float, yA0: float) -> float:
        """Solve the single-dimer extent equation analytically."""
        if kappa <= 0.0 or yA0 <= 0.0:
            return 0.0

        # kappa = alpha*(1 - yA0*alpha/2) / (2*yA0*(1-alpha)^2)
        a = yA0 * (2.0 * kappa + 0.5)
        b = -(4.0 * yA0 * kappa + 1.0)
        c = 2.0 * yA0 * kappa
        disc = max(b * b - 4.0 * a * c, 0.0)
        sqrt_disc = math.sqrt(disc)

        roots = [
            (-b - sqrt_disc) / (2.0 * a),
            (-b + sqrt_disc) / (2.0 * a),
        ]
        valid_roots = [r for r in roots if 0.0 <= r < 1.0]
        if not valid_roots:
            return max(0.0, min(min(roots, key=lambda r: abs(r - 0.5)), 1.0 - 1e-14))
        return min(valid_roots)

    # ------------------------------------------------------------------
    # Dimerization solver
    # ------------------------------------------------------------------

    def solve_dimerization(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        phi_physical: dict[str, float],
    ) -> dict[str, float]:
        """
        Solve 2A ⇌ A₂ for the true equilibrium composition.

        Given nominal mole fractions y_i° and physical fugacity
        coefficients φ_i^P, solves for the extent of dimerization α
        such that:

            n_A  = n_A° (1 - α)
            n_A₂ = n_A° · α / 2
            n_T  = 1 - y_A° · α / 2   (per mole nominal)

        and the equilibrium condition is satisfied.

        Returns true mole fractions y_i (including the dimer).
        """
        y_nominal = self._normalize_nominal_composition(y_nominal)
        K = self.K_eq(T)
        yA0 = y_nominal.get(self.monomer, 0.0)

        # No dimerizing species present
        if yA0 < 1e-12:
            return dict(y_nominal)

        phiA_P = phi_physical.get(self.monomer, 1.0)
        phiA2_P = phi_physical.get(self.dimer, 1.0)

        kappa = K * (P / P_STD) * (phiA_P ** 2) / max(phiA2_P, 1e-30)
        alpha = self._alpha_from_equilibrium(kappa, yA0)

        # True mole fractions
        denom = 1.0 - yA0 * alpha / 2.0
        if denom <= 0:
            alpha = 0.0
            denom = 1.0

        y_true = {}
        for comp, yi0 in y_nominal.items():
            if comp == self.monomer:
                y_true[comp] = yA0 * (1.0 - alpha) / denom
            else:
                y_true[comp] = yi0 / denom  # inerts are diluted

        y_true[self.dimer] = yA0 * alpha / 2.0 / denom

        return y_true

    def _equilibrium_state(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
        max_iter: int = 10,
        tol: float = 1e-8,
        strict_rk: bool = False,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], Optional[str]]:
        """Return true composition, physical phis, total nominal phis, and RK warning text."""
        y_nominal = self._normalize_nominal_composition(y_nominal)
        yA0 = y_nominal.get(self.monomer, 0.0)
        rk_failure = None

        def physical_phi(y_true: dict[str, float]) -> dict[str, float]:
            nonlocal rk_failure
            if rk_model is None:
                return {comp: 1.0 for comp in y_true}
            try:
                phi = rk_model.fugacity_coefficients(T, P, y_true, 'vapor')
            except Exception as exc:
                if strict_rk:
                    raise
                if rk_failure is None:
                    rk_failure = str(exc)
                return {comp: 1.0 for comp in y_true}
            if self.dimer in y_true and self.dimer not in phi:
                phi[self.dimer] = 1.0
            return phi

        if yA0 < 1e-12:
            phi_P = physical_phi(y_nominal)
            phi_total = {comp: max(phi_P.get(comp, 1.0), 1e-30) for comp in y_nominal}
            return y_nominal, phi_P, phi_total, rk_failure

        y_true = dict(y_nominal)
        y_true[self.dimer] = 0.0
        phi_P = {comp: 1.0 for comp in y_true}

        for _ in range(max_iter):
            phi_P = physical_phi(y_true)
            if self.dimer not in phi_P:
                phi_P[self.dimer] = 1.0

            y_new = self.solve_dimerization(T, P, y_nominal, phi_P)
            all_keys = set(y_true.keys()) | set(y_new.keys())
            max_change = max(
                abs(y_new.get(k, 0.0) - y_true.get(k, 0.0))
                for k in all_keys
            )
            y_true = y_new
            if max_change < tol:
                break

        phi_total = {}
        for comp, yi_nom in y_nominal.items():
            phi_P_i = phi_P.get(comp, 1.0)
            yi_true = y_true.get(comp, 0.0)
            phi_D_i = yi_true / yi_nom if yi_nom > 1e-30 else 1.0
            phi_total[comp] = max(phi_P_i * phi_D_i, 1e-30)

        return y_true, phi_P, phi_total, rk_failure

    def _ideal_association_state(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
    ) -> dict:
        """Fast association state for the ideal physical-fugacity path."""
        yA0 = y_nominal.get(self.monomer, 0.0)
        if yA0 < 1e-12:
            y_true = dict(y_nominal)
            phi_P = {comp: 1.0 for comp in y_nominal}
            phi_total = {comp: 1.0 for comp in y_nominal}
        else:
            y_true = self.solve_dimerization(
                T,
                P,
                y_nominal,
                {self.monomer: 1.0, self.dimer: 1.0},
            )
            phi_P = {comp: 1.0 for comp in y_true}
            phi_total = {
                comp: max(y_true.get(comp, 0.0) / yi_nom if yi_nom > 1e-30 else 1.0, 1e-30)
                for comp, yi_nom in y_nominal.items()
            }

        dimer_fraction = max(float(y_true.get(self.dimer, 0.0)), 0.0)
        extent = dimer_fraction / (1.0 + dimer_fraction)
        return {
            "y_true": dict(y_true),
            "phi_physical": dict(phi_P),
            "phi_total": dict(phi_total),
            "extents": {(self.monomer, self.monomer): extent} if extent > 0.0 else {},
            "association_enthalpy": extent * self.association_enthalpy(),
            "rk_failure": None,
        }

    def _association_state_cache_key(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
    ) -> Optional[tuple]:
        if rk_model is not None:
            return None
        return (
            float(T),
            float(P),
            tuple(sorted((comp, float(value)) for comp, value in y_nominal.items())),
        )

    def association_state(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
        max_iter: int = 10,
        tol: float = 1e-8,
        strict_rk: bool = False,
    ) -> dict:
        """Return cached VDM fugacity and association enthalpy state."""
        y_nominal = self._normalize_nominal_composition(y_nominal)
        cache_key = self._association_state_cache_key(T, P, y_nominal, rk_model)
        if cache_key is not None:
            cached = self._association_state_cache.get(cache_key)
            if cached is not None:
                return {
                    **cached,
                    "y_true": dict(cached["y_true"]),
                    "phi_physical": dict(cached["phi_physical"]),
                    "phi_total": dict(cached["phi_total"]),
                    "extents": dict(cached["extents"]),
                }

        if rk_model is None:
            state = self._ideal_association_state(T, P, y_nominal)
        else:
            y_true, phi_P, phi_total, rk_failure = self._equilibrium_state(
                T, P, y_nominal, rk_model, max_iter=max_iter, tol=tol, strict_rk=strict_rk
            )
            dimer_fraction = max(float(y_true.get(self.dimer, 0.0)), 0.0)
            extent = dimer_fraction / (1.0 + dimer_fraction)
            state = {
                "y_true": dict(y_true),
                "phi_physical": dict(phi_P),
                "phi_total": dict(phi_total),
                "extents": {(self.monomer, self.monomer): extent} if extent > 0.0 else {},
                "association_enthalpy": extent * self.association_enthalpy(),
                "rk_failure": rk_failure,
            }
        if cache_key is not None:
            if len(self._association_state_cache) > 20000:
                self._association_state_cache.clear()
            self._association_state_cache[cache_key] = {
                **state,
                "y_true": dict(state["y_true"]),
                "phi_physical": dict(state["phi_physical"]),
                "phi_total": dict(state["phi_total"]),
                "extents": dict(state["extents"]),
            }
        return state

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
        max_iter: int = 10,
        tol: float = 1e-8,
        strict_rk: bool = False,
    ) -> dict[str, float]:
        """
        Total fugacity coefficients:  φ_i = φ_i^P · φ_i^D

        Iterates between RK physical fugacities (at the true composition)
        and the dimerization equilibrium until self-consistent.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [bar].
        y_nominal : dict[str, float]
            Nominal vapor mole fractions — the stoichiometric feed
            composition *without* accounting for dimerization.
        rk_model : RedlichKwong, optional
            RK EOS instance for physical fugacity coefficients.  Must
            include the dimer species in its component list.  If None,
            physical fugacity coefficients default to 1 (ideal gas).
        max_iter : int
            Maximum outer iterations.
        tol : float
            Convergence tolerance on true mole fractions.

        Returns
        -------
        phi : dict[str, float]
            Total fugacity coefficients for each *nominal* component
            (the dimer is not included — it is an internal species).
        """
        state = self.association_state(
            T, P, y_nominal, rk_model, max_iter=max_iter, tol=tol, strict_rk=strict_rk
        )
        rk_failure = state.get("rk_failure")
        if rk_failure is not None:
            warnings.warn(
                f"VDM RK physical fugacity correction failed; using ideal vapor "
                f"physical fugacities. Original error: {rk_failure}",
                RuntimeWarning,
                stacklevel=2,
            )
        return dict(state["phi_total"])

    # ------------------------------------------------------------------
    # Convenience: print diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
    ) -> str:
        """Return a human-readable breakdown of the dimerization state."""
        y_nominal = self._normalize_nominal_composition(y_nominal)
        y_true, phi_P, phi, rk_failure = self._equilibrium_state(
            T, P, y_nominal, rk_model
        )
        K = self.K_eq(T)
        alpha = 0.0
        yA0 = y_nominal.get(self.monomer, 0.0)
        if yA0 > 1e-12 and self.dimer in y_true:
            yA2 = y_true[self.dimer]
            # Recover α from true mole fractions:
            # y_A2 / y_A = (α/2) / (1-α)  →  α = 2·y_A2 / (y_A + 2·y_A2)
            yA = y_true.get(self.monomer, 0.0)
            if yA + 2 * yA2 > 0:
                alpha = 2.0 * yA2 / (yA + 2.0 * yA2)

        lines = [
            f"VDM diagnostics @ T={T:.2f} K, P={P:.4f} bar",
            f"  K_eq = {K:.4f}  (1 bar fugacity standard)",
            f"  alpha = {alpha:.6f}  (fraction dimerized)",
            f"  Nominal composition (stoichiometric):",
        ]
        if rk_failure is not None:
            lines.append(f"  RK physical fugacity fallback: {rk_failure}")
        for comp, yi in sorted(y_nominal.items(), key=lambda x: -x[1]):
            lines.append(f"    {comp:20s} y° = {yi:.6f}")
        lines.append("  True composition (post-dimerization):")
        for comp, yi in sorted(y_true.items(), key=lambda x: -x[1]):
            lines.append(f"    {comp:20s} y  = {yi:.6f}")
        lines.append("  Fugacity coefficients (nominal basis):")
        for comp, phi_i in sorted(phi.items(), key=lambda x: -y_nominal.get(x[0], 0)):
            phi_P_i = phi_P.get(comp, 1.0)
            yi_nom = y_nominal.get(comp, 0.0)
            yi_true = y_true.get(comp, 0.0)
            phi_D_i = yi_true / yi_nom if yi_nom > 1e-30 else 1.0
            lines.append(
                f"    {comp:20s} φ_P={phi_P_i:.6f}  "
                f"φ_D={phi_D_i:.6f}  φ={phi_i:.6f}"
            )

        return "\n".join(lines)


class MultiVaporDimerizationModel:
    """
    Vapor dimerization model for mixtures with multiple associating components.

    Homo-dimer reactions use the supplied per-component VDM models. Cross
    dimers use the physically neutral combining rule:

        ΔH_AB = 0.5*(ΔH_AA + ΔH_BB) + ΔH_residual
        ΔS_AB = 0.5*(ΔS_AA + ΔS_BB) + R*ln(2) + ΔS_residual

    R*ln(2) is the statistical contribution from the two distinguishable
    A-B pairings, giving K_AB = 2*sqrt(K_AA*K_BB) before residuals. Residuals
    default to zero and can be supplied through acid_dimerization.json.
    """

    def __init__(
        self,
        models: dict[str, VaporDimerizationModel],
        cross_residuals: Optional[dict[tuple[str, str], dict[str, float]]] = None,
    ):
        if len(models) < 2:
            raise ValueError(
                "MultiVaporDimerizationModel requires at least two "
                "associating-component models"
            )
        self.models = dict(models)
        self.monomers = tuple(self.models.keys())
        self.cross_residuals = {
            tuple(sorted(pair)): dict(values)
            for pair, values in (cross_residuals or {}).items()
        }
        self._pair_keys = tuple(
            (i, j)
            for index, i in enumerate(self.monomers)
            for j in self.monomers[index:]
        )
        self._dimer_symbols = {
            pair: (self.models[pair[0]].dimer if pair[0] == pair[1] else f"({pair[0]})({pair[1]})")
            for pair in self._pair_keys
        }
        self._dimer_symbol_set = set(self._dimer_symbols.values())
        self._pair_delta_H = {}
        self._pair_delta_S = {}
        for i, j in self._pair_keys:
            if i == j:
                self._pair_delta_H[(i, j)] = self.models[i].association_enthalpy()
                self._pair_delta_S[(i, j)] = self.models[i].association_entropy()
            else:
                delta_H, delta_S = self._cross_delta(i, j)
                self._pair_delta_H[(i, j)] = delta_H
                self._pair_delta_S[(i, j)] = delta_S
        self._fugacity_cache: dict[tuple, dict[str, float]] = {}
        self._association_state_cache: dict[tuple, dict] = {}
        self._compiled_closure_layout_cache: dict[tuple[str, ...], Optional[dict]] = {}

    def _dimer_symbol(self, i: str, j: str) -> str:
        pair = (i, j) if (i, j) in self._dimer_symbols else (j, i)
        return self._dimer_symbols[pair]

    def _cross_delta(self, i: str, j: str) -> tuple[float, float]:
        residual = self.cross_residuals.get(tuple(sorted((i, j))), {})
        delta_H = (
            0.5 * (
                self.models[i].association_enthalpy()
                + self.models[j].association_enthalpy()
            )
            + float(residual.get("delta_H_residual_J_per_mol", 0.0))
        )
        delta_S = (
            0.5 * (
                self.models[i].association_entropy()
                + self.models[j].association_entropy()
            )
            + CROSS_DIMER_STATISTICAL_DELTA_S_J_MOL_K
            + float(residual.get("delta_S_residual_J_per_mol_K", 0.0))
        )
        return delta_H, delta_S

    def _pair_kappa(
        self,
        T: float,
        P: float,
        phi_physical: dict[str, float],
    ) -> dict[tuple[str, str], float]:
        kappa = {}
        for i, j in self._pair_keys:
            K = math.exp(self._pair_delta_S[(i, j)] / R - self._pair_delta_H[(i, j)] / (R * T))
            dimer = self._dimer_symbols[(i, j)]
            phi_i = phi_physical.get(i, 1.0)
            phi_j = phi_physical.get(j, 1.0)
            phi_dimer = max(phi_physical.get(dimer, 1.0), 1e-30)
            kappa[(i, j)] = K * (P / P_STD) * phi_i * phi_j / phi_dimer
        return kappa

    def compiled_vapor_closure(
        self,
        T: float,
        P: float,
        component_order,
        base_values: dict[str, float],
        liquid_composition: Optional[dict[str, float]] = None,
        *,
        max_iter: int,
        tol: float,
        phi_floor: float,
    ) -> Optional[dict]:
        """Run the ideal-physical-fugacity vapor closure in one array kernel."""
        order = tuple(component_order)
        layout = self._compiled_closure_layout(order)
        if layout is None:
            return None

        pressure_factor = float(P) / P_STD
        pair_kappa = [
            math.exp(
                self._pair_delta_S[pair] / R
                - self._pair_delta_H[pair] / (R * float(T))
            ) * pressure_factor
            for pair in self._pair_keys
        ]
        try:
            solved = _compiled_vdm_backend().solve_vapor_closure(
                [max(float(base_values.get(component, 0.0)), 0.0) for component in order],
                [
                    max(float((liquid_composition or {}).get(component, 0.0)), 0.0)
                    for component in order
                ],
                layout['acid_indices'],
                layout['pair_i'],
                layout['pair_j'],
                pair_kappa,
                layout['pair_delta_h'],
                max_iter,
                tol,
                1 if liquid_composition is not None else 0,
                phi_floor,
            )
        except Exception:
            solved = None
        if solved is None:
            return None

        solved['values'] = {
            component: float(solved['values'][index])
            for index, component in enumerate(order)
        }
        solved['vapor_terms'] = {
            component: float(solved['vapor_terms'][index])
            for index, component in enumerate(order)
        }
        solved['vapor_composition'] = {
            component: float(solved['vapor_composition'][index])
            for index, component in enumerate(order)
        }
        solved['phi_total'] = {
            component: float(solved['phi_total'][index])
            for index, component in enumerate(order)
        }
        solved['extents'] = {
            pair: float(solved['extents'][index])
            for index, pair in enumerate(self._pair_keys)
        }
        return solved

    def compiled_association_enthalpy(
        self,
        T: float,
        P: float,
        component_order,
        nominal_composition: dict[str, float],
    ) -> Optional[float]:
        """Return scalar ideal-path association enthalpy without a state dict."""
        order = tuple(component_order)
        layout = self._compiled_closure_layout(order)
        if layout is None:
            return None
        pressure_factor = float(P) / P_STD
        pair_kappa = [
            math.exp(
                self._pair_delta_S[pair] / R
                - self._pair_delta_H[pair] / (R * float(T))
            ) * pressure_factor
            for pair in self._pair_keys
        ]
        try:
            return _compiled_vdm_backend().solve_association_enthalpy(
                [
                    max(float(nominal_composition.get(component, 0.0)), 0.0)
                    for component in order
                ],
                layout['acid_indices'],
                layout['pair_i'],
                layout['pair_j'],
                pair_kappa,
                layout['pair_delta_h'],
            )
        except Exception:
            return None

    def _compiled_closure_layout(self, order: tuple[str, ...]) -> Optional[dict]:
        if not order:
            return None
        layout = self._compiled_closure_layout_cache.get(order)
        if layout is not None or order in self._compiled_closure_layout_cache:
            return layout
        component_index = {component: index for index, component in enumerate(order)}
        if any(component not in component_index for component in self.monomers):
            self._compiled_closure_layout_cache[order] = None
            return None
        monomer_index = {
            component: index for index, component in enumerate(self.monomers)
        }
        layout = {
            'acid_indices': [component_index[component] for component in self.monomers],
            'pair_i': [monomer_index[pair[0]] for pair in self._pair_keys],
            'pair_j': [monomer_index[pair[1]] for pair in self._pair_keys],
            'pair_delta_h': [self._pair_delta_H[pair] for pair in self._pair_keys],
        }
        self._compiled_closure_layout_cache[order] = layout
        return layout

    def _cache_key(self, T: float, P: float, y_nominal: dict[str, float]) -> tuple:
        return (
            float(T),
            float(P),
            tuple((comp, float(y_nominal.get(comp, 0.0))) for comp in self.monomers),
            tuple(
                sorted(
                    (comp, float(value))
                    for comp, value in y_nominal.items()
                    if comp not in self.models and comp not in self._dimer_symbol_set
                )
            ),
        )

    def _ideal_phi_fast_fugacity_coefficients(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
    ) -> dict[str, float]:
        return dict(self.association_state(T, P, y_nominal, rk_model=None)["phi_total"])

    def _ideal_association_state(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
    ) -> dict:
        key = self._cache_key(T, P, y_nominal)
        cached = self._association_state_cache.get(key)
        if cached is not None:
            return {
                **cached,
                "y_true": dict(cached["y_true"]),
                "phi_total": dict(cached["phi_total"]),
                "extents": dict(cached["extents"]),
            }
        y_nominal = _normalize_nominal_composition_for_dimers(
            y_nominal,
            self._dimer_symbol_set,
        )
        kappa = {}
        pressure_factor = P / P_STD
        for pair in self._pair_keys:
            K = math.exp(self._pair_delta_S[pair] / R - self._pair_delta_H[pair] / (R * T))
            kappa[pair] = K * pressure_factor
        monomer_moles, extents, total = self._solve_true_moles(y_nominal, kappa)

        phi_total = {}
        for comp, yi_nom in y_nominal.items():
            if comp in self.models:
                yi_true = monomer_moles.get(comp, 0.0) / max(total, 1e-300)
                phi_total[comp] = float(max(yi_true / yi_nom if yi_nom > 1e-30 else 1.0, 1e-30))
            else:
                phi_total[comp] = 1.0
        y_true = {
            comp: max(y_nominal.get(comp, 0.0), 0.0) / total
            for comp in y_nominal
            if comp not in self.monomers
        }
        for comp in self.monomers:
            if y_nominal.get(comp, 0.0) > 0.0:
                y_true[comp] = monomer_moles.get(comp, 0.0) / total
        for pair, extent in extents.items():
            if extent > 0.0:
                y_true[self._dimer_symbols[pair]] = extent / total
        total_y = sum(y_true.values())
        if total_y > 0.0:
            y_true = {comp: value / total_y for comp, value in y_true.items()}
        state = {
            "y_true": dict(y_true),
            "phi_total": dict(phi_total),
            "extents": dict(extents),
            "association_enthalpy": sum(
                extent * self._pair_delta_H[pair]
                for pair, extent in extents.items()
            ),
            "rk_failure": None,
        }
        if len(self._association_state_cache) > 20000:
            self._association_state_cache.clear()
        self._association_state_cache[key] = {
            **state,
            "y_true": dict(state["y_true"]),
            "phi_total": dict(state["phi_total"]),
            "extents": dict(state["extents"]),
        }
        self._fugacity_cache[key] = dict(phi_total)
        return state

    def association_state(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
        max_iter: int = 10,
        tol: float = 1e-8,
        strict_rk: bool = False,
    ) -> dict:
        """Return VDM fugacity and association enthalpy state."""
        if rk_model is None:
            return self._ideal_association_state(T, P, y_nominal)

        y_nominal = _normalize_nominal_composition_for_dimers(
            y_nominal,
            self._dimer_symbol_set,
        )
        rk_failure = None

        def physical_phi(y_true: dict[str, float]) -> dict[str, float]:
            nonlocal rk_failure
            try:
                return rk_model.fugacity_coefficients(T, P, y_true, 'vapor')
            except Exception as exc:
                if strict_rk:
                    raise
                if rk_failure is None:
                    rk_failure = str(exc)
                return {comp: 1.0 for comp in y_true}

        y_true = dict(y_nominal)
        phi_P = {comp: 1.0 for comp in y_true}
        extents = {}
        for _ in range(max_iter):
            phi_P = physical_phi(y_true)
            kappa = self._pair_kappa(T, P, phi_P)
            _monomers, extents, _total = self._solve_true_moles(y_nominal, kappa)
            y_new = self.solve_dimerization(T, P, y_nominal, phi_P)
            keys = set(y_true) | set(y_new)
            max_change = max(abs(y_new.get(key, 0.0) - y_true.get(key, 0.0)) for key in keys)
            y_true = y_new
            if max_change < tol:
                break

        phi_total = {}
        for comp, yi_nom in y_nominal.items():
            phi_P_i = phi_P.get(comp, 1.0)
            yi_true = y_true.get(comp, 0.0)
            phi_D_i = yi_true / yi_nom if yi_nom > 1e-30 else 1.0
            phi_total[comp] = float(max(phi_P_i * phi_D_i, 1e-30))
        return {
            "y_true": dict(y_true),
            "phi_total": dict(phi_total),
            "extents": dict(extents),
            "association_enthalpy": sum(
                extent * self._pair_delta_H[pair]
                for pair, extent in extents.items()
            ),
            "rk_failure": rk_failure,
        }

    def _solve_true_moles(
        self,
        y_nominal: dict[str, float],
        kappa: dict[tuple[str, str], float],
    ) -> tuple[dict[str, float], dict[tuple[str, str], float], float]:
        n0 = {comp: max(float(y_nominal.get(comp, 0.0)), 0.0) for comp in self.monomers}
        inerts = {
            comp: max(float(value), 0.0)
            for comp, value in y_nominal.items()
            if comp not in n0
        }
        inert_total = sum(inerts.values())
        acid_components = [comp for comp in self.monomers if n0.get(comp, 0.0) > 1e-14]
        if not acid_components:
            return {}, {}, 1.0

        if len(acid_components) == 2:
            solution = self._solve_true_moles_two_acids(acid_components, n0, inert_total, kappa)
            if solution is not None:
                return solution

        solution = self._solve_true_moles_newton(acid_components, n0, inert_total, kappa)
        if solution is not None:
            return solution

        return self._solve_true_moles_least_squares(acid_components, n0, inert_total, kappa)

    def _solve_true_moles_newton(
        self,
        acid_components: list[str],
        n0: dict[str, float],
        inert_total: float,
        kappa: dict[tuple[str, str], float],
    ) -> Optional[tuple[dict[str, float], dict[tuple[str, str], float], float]]:
        n = len(acid_components)
        if n < 2:
            return None

        pair_i_list = []
        pair_j_list = []
        pair_kappa_list = []
        pair_keys = []
        for i, comp_i in enumerate(acid_components):
            for j, comp_j in enumerate(acid_components[i:], start=i):
                pair_i_list.append(i)
                pair_j_list.append(j)
                pair_kappa_list.append(max(kappa.get((comp_i, comp_j), 0.0), 0.0))
                pair_keys.append((comp_i, comp_j))

        try:
            compiled = _compiled_vdm_backend().solve_n_acid_true_moles(
                [max(n0.get(comp, 0.0), 0.0) for comp in acid_components],
                inert_total,
                pair_i_list,
                pair_j_list,
                pair_kappa_list,
            )
        except Exception:
            compiled = None
        if compiled is not None:
            monomers, extents_list, total = compiled
            monomer_moles = {comp: 0.0 for comp in self.monomers}
            for pos, comp in enumerate(acid_components):
                monomer_moles[comp] = float(max(monomers[pos], 0.0))
            extents = {
                pair_keys[pos]: float(max(extents_list[pos], 0.0))
                for pos in range(len(pair_keys))
            }
            return monomer_moles, extents, max(float(total), 1e-300)

        import numpy as np

        nominal = np.array(
            [max(n0.get(comp, 0.0), 0.0) for comp in acid_components],
            dtype=float,
        )
        if np.any(nominal <= 0.0):
            return None
        scales = np.maximum(nominal, 1e-12)
        pair_i_array = np.array(pair_i_list, dtype=int)
        pair_j_array = np.array(pair_j_list, dtype=int)
        pair_kappa_array = np.array(pair_kappa_list, dtype=float)

        def evaluate(monomers: np.ndarray):
            terms = pair_kappa_array * monomers[pair_i_array] * monomers[pair_j_array]
            association_sum = float(np.sum(terms))
            base_total = float(inert_total + np.sum(monomers))
            root = math.sqrt(max(base_total * base_total + 4.0 * association_sum, 0.0))
            if root <= 1e-300:
                return None
            total = max(0.5 * (base_total + root), 1e-300)
            extents_array = np.maximum(terms / total, 0.0)
            residual = monomers - nominal
            for pos, extent in enumerate(extents_array):
                i = pair_i_array[pos]
                j = pair_j_array[pos]
                if i == j:
                    residual[i] += 2.0 * extent
                else:
                    residual[i] += extent
                    residual[j] += extent
            residual = residual / scales

            d_assoc = np.zeros(n, dtype=float)
            d_terms = np.zeros((len(pair_keys), n), dtype=float)
            for pos, coeff in enumerate(pair_kappa_array):
                i = pair_i_array[pos]
                j = pair_j_array[pos]
                if i == j:
                    derivative = 2.0 * coeff * monomers[i]
                    d_terms[pos, i] = derivative
                    d_assoc[i] += derivative
                else:
                    d_i = coeff * monomers[j]
                    d_j = coeff * monomers[i]
                    d_terms[pos, i] = d_i
                    d_terms[pos, j] = d_j
                    d_assoc[i] += d_i
                    d_assoc[j] += d_j

            d_total = 0.5 * (1.0 + (base_total + 2.0 * d_assoc) / root)
            inv_total_sq = 1.0 / (total * total)
            jacobian = np.eye(n, dtype=float)
            for pos, term in enumerate(terms):
                i = pair_i_array[pos]
                j = pair_j_array[pos]
                for col in range(n):
                    d_extent = (d_terms[pos, col] * total - term * d_total[col]) * inv_total_sq
                    if i == j:
                        jacobian[i, col] += 2.0 * d_extent
                    else:
                        jacobian[i, col] += d_extent
                        jacobian[j, col] += d_extent
            jacobian = jacobian / scales[:, None]
            extents = {
                pair_keys[pos]: float(extents_array[pos])
                for pos in range(len(pair_keys))
            }
            return residual, jacobian, extents, total

        monomers = np.maximum(0.5 * nominal, 1e-300)
        best = None
        best_norm = math.inf
        for _ in range(40):
            evaluated = evaluate(monomers)
            if evaluated is None:
                return None
            residual, jacobian, extents, total = evaluated
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if norm < best_norm:
                best = (monomers.copy(), extents, total)
                best_norm = norm
            if norm < 1e-11:
                monomer_moles = {comp: 0.0 for comp in self.monomers}
                for pos, comp in enumerate(acid_components):
                    monomer_moles[comp] = float(max(monomers[pos], 0.0))
                return monomer_moles, extents, total

            try:
                delta = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break

            accepted = False
            for attempt in range(24):
                step = 0.5 ** attempt
                trial = monomers + step * delta
                if np.any(trial <= 1e-300) or np.any(trial > nominal):
                    continue
                trial_evaluated = evaluate(trial)
                if trial_evaluated is None:
                    continue
                trial_residual = trial_evaluated[0]
                trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
                if trial_norm < norm:
                    monomers = trial
                    accepted = True
                    break
            if not accepted:
                break

        if best is not None and best_norm < 1e-8:
            monomers, extents, total = best
            monomer_moles = {comp: 0.0 for comp in self.monomers}
            for pos, comp in enumerate(acid_components):
                monomer_moles[comp] = float(max(monomers[pos], 0.0))
            return monomer_moles, extents, total
        return None

    def _solve_true_moles_least_squares(
        self,
        acid_components: list[str],
        n0: dict[str, float],
        inert_total: float,
        kappa: dict[tuple[str, str], float],
    ) -> tuple[dict[str, float], dict[tuple[str, str], float], float]:
        from scipy.optimize import least_squares

        def extents_from_monomers(monomer_moles: dict[str, float]):
            base_total = inert_total + sum(monomer_moles.values())
            association_sum = 0.0
            pair_terms = {}
            for index, i in enumerate(acid_components):
                for j in acid_components[index:]:
                    term = kappa.get((i, j), 0.0) * monomer_moles[i] * monomer_moles[j]
                    pair_terms[(i, j)] = max(term, 0.0)
                    association_sum += max(term, 0.0)
            total = 0.5 * (
                base_total
                + math.sqrt(max(base_total * base_total + 4.0 * association_sum, 0.0))
            )
            total = max(total, 1e-300)
            extents = {pair: term / total for pair, term in pair_terms.items()}
            return extents, total

        def residual(values):
            monomer_moles = {
                comp: max(values[pos], 1e-300)
                for pos, comp in enumerate(acid_components)
            }
            extents, _total = extents_from_monomers(monomer_moles)
            result = []
            for comp in acid_components:
                consumed = 0.0
                for (i, j), extent in extents.items():
                    if i == comp and j == comp:
                        consumed += 2.0 * extent
                    elif i == comp or j == comp:
                        consumed += extent
                scale = max(n0[comp], 1e-12)
                result.append((monomer_moles[comp] + consumed - n0[comp]) / scale)
            return result

        x0 = [max(n0[comp] * 0.5, 1e-14) for comp in acid_components]
        upper = [max(n0[comp], 1e-14) for comp in acid_components]
        solution = least_squares(
            residual,
            x0,
            bounds=([1e-300] * len(acid_components), upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )
        monomer_moles = {
            comp: max(solution.x[pos], 0.0)
            for pos, comp in enumerate(acid_components)
        }
        for comp in self.monomers:
            monomer_moles.setdefault(comp, 0.0)
        extents, total = extents_from_monomers(monomer_moles)
        return monomer_moles, extents, total

    def _solve_true_moles_two_acids(
        self,
        acid_components: list[str],
        n0: dict[str, float],
        inert_total: float,
        kappa: dict[tuple[str, str], float],
    ) -> Optional[tuple[dict[str, float], dict[tuple[str, str], float], float]]:
        comp_a, comp_b = acid_components
        n_a = max(n0.get(comp_a, 0.0), 0.0)
        n_b = max(n0.get(comp_b, 0.0), 0.0)
        if n_a <= 0.0 or n_b <= 0.0:
            return None

        k_aa = max(kappa.get((comp_a, comp_a), 0.0), 0.0)
        k_ab = max(kappa.get((comp_a, comp_b), kappa.get((comp_b, comp_a), 0.0)), 0.0)
        k_bb = max(kappa.get((comp_b, comp_b), 0.0), 0.0)
        scale_a = max(n_a, 1e-12)
        scale_b = max(n_b, 1e-12)

        try:
            compiled = _compiled_vdm_backend().solve_two_acid_true_moles(
                n_a, n_b, inert_total, k_aa, k_ab, k_bb
            )
        except Exception:
            compiled = None
        if compiled is not None:
            a, b, e_aa, e_ab, e_bb, total = compiled
            monomer_moles = {comp: 0.0 for comp in self.monomers}
            monomer_moles[comp_a] = max(a, 0.0)
            monomer_moles[comp_b] = max(b, 0.0)
            extents = {
                (comp_a, comp_a): max(e_aa, 0.0),
                (comp_a, comp_b): max(e_ab, 0.0),
                (comp_b, comp_b): max(e_bb, 0.0),
            }
            return monomer_moles, extents, max(total, 1e-300)

        def evaluate(a: float, b: float):
            term_aa = k_aa * a * a
            term_ab = k_ab * a * b
            term_bb = k_bb * b * b
            association_sum = term_aa + term_ab + term_bb
            base_total = inert_total + a + b
            disc = base_total * base_total + 4.0 * association_sum
            root = math.sqrt(max(disc, 0.0))
            total = max(0.5 * (base_total + root), 1e-300)

            e_aa = term_aa / total
            e_ab = term_ab / total
            e_bb = term_bb / total
            f_a = a + 2.0 * e_aa + e_ab - n_a
            f_b = b + 2.0 * e_bb + e_ab - n_b

            if root <= 1e-300:
                return None

            d_assoc_da = 2.0 * k_aa * a + k_ab * b
            d_assoc_db = k_ab * a + 2.0 * k_bb * b
            d_total_da = 0.5 * (1.0 + (base_total + 2.0 * d_assoc_da) / root)
            d_total_db = 0.5 * (1.0 + (base_total + 2.0 * d_assoc_db) / root)
            inv_total_sq = 1.0 / (total * total)

            def d_extent(term: float, d_term: float, d_total: float) -> float:
                return (d_term * total - term * d_total) * inv_total_sq

            de_aa_da = d_extent(term_aa, 2.0 * k_aa * a, d_total_da)
            de_aa_db = d_extent(term_aa, 0.0, d_total_db)
            de_ab_da = d_extent(term_ab, k_ab * b, d_total_da)
            de_ab_db = d_extent(term_ab, k_ab * a, d_total_db)
            de_bb_da = d_extent(term_bb, 0.0, d_total_da)
            de_bb_db = d_extent(term_bb, 2.0 * k_bb * b, d_total_db)

            jacobian = (
                ((1.0 + 2.0 * de_aa_da + de_ab_da) / scale_a,
                 (2.0 * de_aa_db + de_ab_db) / scale_a),
                ((2.0 * de_bb_da + de_ab_da) / scale_b,
                 (1.0 + 2.0 * de_bb_db + de_ab_db) / scale_b),
            )
            residual = (f_a / scale_a, f_b / scale_b)
            extents = {
                (comp_a, comp_a): max(e_aa, 0.0),
                (comp_a, comp_b): max(e_ab, 0.0),
                (comp_b, comp_b): max(e_bb, 0.0),
            }
            return residual, jacobian, extents, total

        a = min(max(0.5 * n_a, 1e-300), n_a)
        b = min(max(0.5 * n_b, 1e-300), n_b)
        best = None
        best_norm = math.inf

        for _ in range(30):
            evaluated = evaluate(a, b)
            if evaluated is None:
                return None
            residual, jacobian, extents, total = evaluated
            norm = max(abs(residual[0]), abs(residual[1]))
            if norm < best_norm:
                best = (a, b, extents, total)
                best_norm = norm
            if norm < 1e-11:
                monomer_moles = {comp: 0.0 for comp in self.monomers}
                monomer_moles[comp_a] = max(a, 0.0)
                monomer_moles[comp_b] = max(b, 0.0)
                return monomer_moles, extents, total

            (j11, j12), (j21, j22) = jacobian
            det = j11 * j22 - j12 * j21
            if not math.isfinite(det) or abs(det) < 1e-30:
                break
            delta_a = (-residual[0] * j22 + j12 * residual[1]) / det
            delta_b = (j21 * residual[0] - j11 * residual[1]) / det
            if not (math.isfinite(delta_a) and math.isfinite(delta_b)):
                break

            accepted = False
            for attempt in range(20):
                step = 0.5 ** attempt
                trial_a = a + step * delta_a
                trial_b = b + step * delta_b
                if not (1e-300 < trial_a <= n_a and 1e-300 < trial_b <= n_b):
                    continue
                trial = evaluate(trial_a, trial_b)
                if trial is None:
                    continue
                trial_residual, _trial_jacobian, _trial_extents, _trial_total = trial
                trial_norm = max(abs(trial_residual[0]), abs(trial_residual[1]))
                if trial_norm < norm:
                    a, b = trial_a, trial_b
                    accepted = True
                    break
            if not accepted:
                break

        if best is not None and best_norm < 1e-8:
            a, b, extents, total = best
            monomer_moles = {comp: 0.0 for comp in self.monomers}
            monomer_moles[comp_a] = max(a, 0.0)
            monomer_moles[comp_b] = max(b, 0.0)
            return monomer_moles, extents, total
        return None

    def solve_dimerization(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        phi_physical: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        y_nominal = _normalize_nominal_composition_for_dimers(
            y_nominal,
            self._dimer_symbol_set,
        )
        phi_physical = phi_physical or {}
        kappa = self._pair_kappa(T, P, phi_physical)
        monomer_moles, extents, total = self._solve_true_moles(y_nominal, kappa)

        y_true = {
            comp: max(y_nominal.get(comp, 0.0), 0.0) / total
            for comp in y_nominal
            if comp not in self.monomers
        }
        for comp in self.monomers:
            if y_nominal.get(comp, 0.0) > 0.0:
                y_true[comp] = monomer_moles.get(comp, 0.0) / total
        for (i, j), extent in extents.items():
            if extent > 0.0:
                y_true[self._dimer_symbols[(i, j)]] = extent / total

        total_y = sum(y_true.values())
        if total_y > 0:
            y_true = {comp: value / total_y for comp, value in y_true.items()}
        return y_true

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        y_nominal: dict[str, float],
        rk_model=None,
        max_iter: int = 10,
        tol: float = 1e-8,
        strict_rk: bool = False,
    ) -> dict[str, float]:
        state = self.association_state(
            T, P, y_nominal, rk_model, max_iter=max_iter, tol=tol, strict_rk=strict_rk
        )
        rk_failure = state.get("rk_failure")
        if rk_failure is not None:
            warnings.warn(
                f"VDM RK physical fugacity correction failed; using ideal vapor "
                f"physical fugacities. Original error: {rk_failure}",
                RuntimeWarning,
                stacklevel=2,
            )
        return dict(state["phi_total"])


# ------------------------------------------------------------------
# Dimer property estimation
# ------------------------------------------------------------------

def estimate_dimer_properties(
    monomer_symbol: str,
    dimer_symbol: str,
    db=None,
) -> dict:
    """
    Estimate critical properties and other parameters for a dimer.

    Returns a dict suitable for creating a ChemicalProperties entry
    or adding to the chemicals database.

    Heuristics (Prausnitz et al., Properties of Gases and Liquids):
      MW_dimer  = 2 × MW_monomer
      Tc_dimer  ≈ 1.4–1.6 × Tc_monomer
      Pc_dimer  ≈ 0.25–0.4 × Pc_monomer
      ω_dimer   ≈ 1.5–2 × ω_monomer
      Tb_dimer  ≈ 1.2–1.4 × Tb_monomer
      Cp_dimer  ≈ 2 × Cp_monomer (additive)
    """
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .chemical_properties import get_database
    else:
        from chemical_properties import get_database

    if db is None:
        db = get_database()

    mono = db.get(monomer_symbol)
    if mono is None:
        raise ValueError(
            f"Monomer '{monomer_symbol}' not found in database; "
            f"cannot estimate dimer properties."
        )

    params = get_dimerization_params(monomer_symbol)
    delta_H_kJ_per_mol = (
        params["delta_H_J_per_mol"] / 1000.0
        if params is not None
        else -60.0
    )
    Hf = None
    if mono.Hf is not None:
        # ChemicalProperties.Hf is kJ/mol.  Dimerization enthalpy is for
        # 2 monomers -> 1 dimer, so add it directly to 2*Hf_monomer.
        Hf = 2.0 * mono.Hf + delta_H_kJ_per_mol

    return {
        "symbol": dimer_symbol,
        "name": f"{mono.name} dimer",
        "formula": f"({mono.formula})2",
        "MW": mono.MW * 2.0,
        "Tc": mono.Tc * 1.5 if mono.Tc else None,
        "Pc": mono.Pc * 0.3 if mono.Pc else None,
        "omega": mono.omega * 2.0 if mono.omega is not None else None,
        "Tb": mono.Tb * 1.3 if mono.Tb else None,
        "Hf": Hf,
        "Cp_coeffs": [c * 2.0 for c in mono.Cp_coeffs] if mono.Cp_coeffs else [66, 0, 0, 0],
        "phase_at_STP": "gas",
        "Hvap": mono.Hvap * 1.5 if mono.Hvap else None,
        "source": "vdm_estimated",
        "notes": "Estimated dimer critical properties are rough and should not be used for high-pressure design.",
    }


def build_augmented_rk(
    monomer: str,
    dimer: str,
    other_components: list[str],
    db=None,
) -> tuple:
    """
    Build an RK model that includes the dimer species.

    Adds an estimated dimer entry to the database, then creates a
    RedlichKwong instance that covers all species including the dimer.

    Returns
    -------
    (rk_model, db) : (RedlichKwong, ChemicalDatabase)
        The RK model and the (possibly modified) database.
    """
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .rk_eos import RedlichKwong
    else:
        from rk_eos import RedlichKwong
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .chemical_properties import ChemicalProperties, get_database
    else:
        from chemical_properties import ChemicalProperties, get_database

    if db is None:
        db = get_database()

    # Ensure dimer is in the database
    if db.get(dimer) is None:
        dimer_props = estimate_dimer_properties(monomer, dimer, db)
        entry = ChemicalProperties(
            symbol=dimer_props["symbol"],
            name=dimer_props["name"],
            formula=dimer_props["formula"],
            MW=dimer_props["MW"],
            Tc=dimer_props["Tc"],
            Pc=dimer_props["Pc"],
            omega=dimer_props["omega"],
            Tb=dimer_props["Tb"],
            Hf=dimer_props["Hf"],
            Cp_coeffs=dimer_props["Cp_coeffs"],
            Hvap=dimer_props["Hvap"],
            phase_at_STP=dimer_props["phase_at_STP"],
            source=dimer_props["source"],
        )
        db.chemicals[dimer] = entry

    all_components = list(dict.fromkeys([*other_components, monomer, dimer]))
    rk = RedlichKwong(all_components, db)
    return rk, db


# ------------------------------------------------------------------
# Database lookup
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _dimerization_data() -> dict:
    """Load the acid dimerization JSON database (cached)."""
    import json

    path = DATA_DIR / "acid_dimerization.json"
    return json.loads(path.read_text())


def _lookup_values_for_identifier(identifier: str) -> set[str]:
    """Return conservative lookup keys for an acid identifier."""
    identifier_text = str(identifier)
    candidates = [identifier]
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .compound_identity import get_compound_identity_resolver, looks_like_formula
        else:
            from compound_identity import get_compound_identity_resolver, looks_like_formula

        # Brute formulas such as C3H6O2 are structurally ambiguous, so do not
        # let the general identity resolver turn them into a carboxylic acid.
        formula_like = looks_like_formula(identifier_text) and any(
            ch.isdigit() for ch in identifier_text
        )
        if not (formula_like and "COOH" not in identifier_text.upper()):
            candidates = get_compound_identity_resolver().candidate_identifiers(identifier)
    except Exception:
        pass

    values = set()
    for value in candidates:
        if not value:
            continue
        values.update(_lookup_keys(value))
    return values


def _lookup_keys(value: str) -> set[str]:
    """Return normalized lookup keys for names, formulas, CAS, and ChemSep ids."""
    raw = str(value).strip()
    if not raw:
        return set()
    keys = {raw, raw.lower()}
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .compound_identity import compact_identifier, normalize_formula, normalize_name
        else:
            from compound_identity import compact_identifier, normalize_formula, normalize_name

        keys.add(normalize_name(raw))
        keys.add(compact_identifier(raw))
        keys.add(normalize_formula(raw))
    except Exception:
        import re

        keys.add(re.sub(r"[^a-z0-9]+", "", raw.lower()))
        keys.add(re.sub(r"[^A-Za-z0-9()]+", "", raw).upper())
    return {key for key in keys if key}


def _dimerization_entry_keys(entry: dict) -> set[str]:
    keys = set()
    for field in ("symbol", "name", "formula", "CAS", "chemsep_id"):
        value = entry.get(field)
        if value:
            keys.update(_lookup_keys(str(value)))
    for alias in entry.get("aliases", []):
        keys.update(_lookup_keys(str(alias)))
    return keys


def _dimerization_entry_for_identifier(identifier: str) -> Optional[dict]:
    lookup_keys = _lookup_values_for_identifier(identifier)
    for entry in _dimerization_data().get("dimers", []):
        if lookup_keys & _dimerization_entry_keys(entry):
            return entry
    return None


def is_monocarboxylic_acid(
    identifier: str,
    *,
    smiles: Optional[str] = None,
) -> bool:
    """Return whether identity evidence establishes exactly one COOH group."""
    if _dimerization_entry_for_identifier(identifier) is not None:
        return True
    if _is_saturated_carboxylic_acid(identifier):
        return True

    identifier_text = str(identifier)
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .compound_identity import looks_like_formula
        else:
            from compound_identity import looks_like_formula

        formula_like = looks_like_formula(identifier_text) and any(
            character.isdigit()
            for character in identifier_text
        )
    except Exception:
        formula_like = False
    if (
        formula_like
        and "COOH" not in identifier_text.upper()
        and smiles is None
    ):
        return False

    identifier_parts = identifier_text.split("-")
    cas_like = (
        len(identifier_parts) == 3
        and all(part.isdigit() for part in identifier_parts)
    )
    smiles_candidates = [smiles]
    if smiles is None and not cas_like:
        smiles_candidates.append(identifier_text)
    if smiles is None:
        try:
            from chemicals.identifiers import search_chemical

            metadata = search_chemical(identifier_text)
            smiles_candidates.append(metadata.smiles)
        except Exception:
            pass
    try:
        from rdkit import Chem

        carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        if carboxyl is None:
            return False
        for candidate in smiles_candidates:
            if not candidate:
                continue
            molecule = Chem.MolFromSmiles(str(candidate))
            if (
                molecule is not None
                and len(molecule.GetSubstructMatches(carboxyl)) == 1
            ):
                return True
    except ImportError:
        return False
    return False


def get_dimerization_params(symbol: str) -> Optional[dict]:
    """
    Look up dimerization parameters for a carboxylic acid.

    Returns a dict with keys ``delta_S_J_per_mol_K`` and
    ``delta_H_J_per_mol``.  If the acid is not in the database but is
    recognised as a saturated carboxylic acid (CₙH₂ₙ₊₁COOH), a default
    estimate is returned (ΔH = −60.5 kJ/mol, ΔS = −144 J/mol·K).
    Returns None for non-acids.
    """
    entry = _dimerization_entry_for_identifier(symbol)
    if entry is not None:
        return {
            "delta_S_J_per_mol_K": entry["delta_S"],
            "delta_H_J_per_mol": entry["delta_H"],
            "name": entry.get("name", ""),
            "formula": entry.get("formula", ""),
            "source": "database",
        }

    if is_monocarboxylic_acid(symbol):
        return {
            "delta_S_J_per_mol_K": GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
            "delta_H_J_per_mol": GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
            "name": symbol,
            "formula": symbol,
            "source": "estimated (monocarboxylic acid default)",
        }

    return None


def get_cross_dimerization_residual(symbol1: str, symbol2: str) -> dict[str, float]:
    """
    Return residual cross-association corrections for a pair of acids.

    The JSON entry is optional.  If absent, the cross dimer uses the ideal
    combining rule:

        ΔH_AB = 0.5*(ΔH_AA + ΔH_BB)
        ΔS_AB = 0.5*(ΔS_AA + ΔS_BB) + R*ln(2)

    The entropy term is the statistical factor for two distinguishable A-B
    pairings, so K_AB = 2*sqrt(K_AA*K_BB) before residual corrections.
    """
    lookup1 = _lookup_values_for_identifier(symbol1)
    lookup2 = _lookup_values_for_identifier(symbol2)
    data = _dimerization_data()
    for entry in data.get("cross_dimers", []):
        values = entry.get("symbols") or entry.get("components") or entry.get("pair")
        if not values or len(values) != 2:
            continue
        entry1 = _lookup_keys(str(values[0]))
        entry2 = _lookup_keys(str(values[1]))
        if (
            (lookup1 & entry1 and lookup2 & entry2)
            or (lookup1 & entry2 and lookup2 & entry1)
        ):
            return {
                "delta_H_residual_J_per_mol": float(
                    entry.get("delta_H_residual", entry.get("delta_H_residual_J_per_mol", 0.0))
                ),
                "delta_S_residual_J_per_mol_K": float(
                    entry.get("delta_S_residual", entry.get("delta_S_residual_J_per_mol_K", 0.0))
                ),
                "source": entry.get("source", "database"),
            }
    return {
        "delta_H_residual_J_per_mol": 0.0,
        "delta_S_residual_J_per_mol_K": 0.0,
        "source": "ideal combining rule",
    }


def _is_saturated_carboxylic_acid(symbol: str) -> bool:
    """
    Check whether *symbol* represents a saturated carboxylic acid
    of the form CₙH₂ₙ₊₁COOH  (contains only C, H; O only in -COOH).
    """
    import re

    s = symbol.strip()

    # --- Must contain the -COOH structural marker ---
    # This distinguishes acids from esters (CH3COOC2H5) and brute
    # formulas (C4H8O2) which are atom-count-ambiguous.
    if re.search(r'COOH', s, re.IGNORECASE):
        atoms = _parse_formula(s)
        if atoms is not None:
            return _check_saturated_acid(atoms)

    # --- Common acid names (no COOH in the text) ---
    name_lower = s.lower().replace(' ', '').replace('-', '').replace('_', '')
    known = {
        'formicacid': 'HCOOH',
        'aceticacid': 'CH3COOH',
        'propionicacid': 'C2H5COOH',
        'nbutyricacid': 'C3H7COOH',
        'butyricacid': 'C3H7COOH',
        'isobutyricacid': '(CH3)2CHCOOH',
        'valericacid': 'C4H9COOH',
        'nvalericacid': 'C4H9COOH',
        'hexanoicacid': 'C5H11COOH',
        'heptanoicacid': 'C6H13COOH',
        'octanoicacid': 'C7H15COOH',
        'nonanoicacid': 'C8H17COOH',
        'decanoicacid': 'C9H19COOH',
    }
    formula = known.get(name_lower)
    if formula is not None:
        atoms = _parse_formula(formula)
        if atoms is not None:
            return _check_saturated_acid(atoms)

    return False


def _parse_formula(formula: str) -> Optional[dict[str, int]]:
    """Parse a chemical formula string into atom counts.

    Handles simple organics:  CH3COOH, C2H5COOH, C3H7COOH, HCOOH
    Also handles parenthetical groups: (CH3)2CHCOOH
    Returns None if the formula can't be parsed.
    """
    import re

    formula = formula.strip()
    if not formula:
        return None

    # Remove any parenthetical multiplication by expanding:
    # (CH3)2CHCOOH → CH3CH3CHCOOH
    while '(' in formula:
        formula = re.sub(
            r'\(([A-Za-z0-9]+)\)(\d+)',
            lambda m: m.group(1) * int(m.group(2)),
            formula,
        )
        if '(' in formula:
            # Prevent infinite loop on malformed input
            break

    # Parse flat formula: uppercase letter + optional lowercase + optional digits
    counts: dict[str, int] = {}
    for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        el = m.group(1)
        n = int(m.group(2)) if m.group(2) else 1
        counts[el] = counts.get(el, 0) + n

    return counts if counts else None


def _check_saturated_acid(atoms: dict[str, int]) -> bool:
    """Given atom counts, check for CₙH₂ₙ₊₁COOH (saturated, only C/H/O)."""
    c = atoms.get('C', 0)
    h = atoms.get('H', 0)
    o = atoms.get('O', 0)

    # Must have C, H, and exactly 2 O (the -COOH group)
    if c < 1 or h < 1 or o != 2:
        return False

    # Must contain only C, H, O
    if set(atoms.keys()) - {'C', 'H', 'O'}:
        return False

    # Saturated check:  H = 2C + 2  for CₙH₂ₙ₊₁COOH
    # Total formula is CₙH₂ₙ₊₁COOH = C_{n+1} H_{2n+2} O₂
    # Let m = n+1 = total carbons. Then H = 2m
    # Wait: C_n H_{2n+1} COOH gives total C = n+1, total H = 2n+1+1 = 2n+2 = 2(n+1)
    # So H = 2C for a saturated monoacid.
    if h != 2 * c:
        return False

    return True


def list_dimerizing_acids() -> list[dict]:
    """Return all entries in the dimerization database."""
    data = _dimerization_data()
    return [
        {
            "symbol": e["symbol"],
            "name": e["name"],
            "formula": e.get("formula", ""),
            "delta_S_J_per_mol_K": e["delta_S"],
            "delta_H_J_per_mol": e["delta_H"],
        }
        for e in data.get("dimers", [])
    ]
