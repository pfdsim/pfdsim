"""Lee-Kesler-Plocker phi-phi thermodynamics using DWSIM data/correlations.

Equations: Plocker, Knapp & Prausnitz (1978), doi:10.1021/i260067a020.
Estimation rules: https://dwsim.org/help/main/70-lee-kesler-plocker.html
All departures and component fugacities derive from the same residual Gibbs
energy. T/P use K/bar; volumes m3/kmol; caloric properties kJ/kmol and K.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

if __package__.split('.', 1)[0] == 'pfdsim':
    from ..phi_phi import PhiPhiFlashError, normalized_eos_composition
else:
    from phi_phi import PhiPhiFlashError, normalized_eos_composition

from .common import R, ThermodynamicsError, ThermodynamicsValidityError
from .eos import PhiPhiThermodynamics, _phi_henry_aqueous_K_values

# b1..b4, c1..c4, d1,d2,beta,gamma, from DWSIM's Z_LK.
LK_CONSTANTS = (
    (0.1181193, 0.265728, 0.15479, 0.030323,
     0.0236744, 0.0186984, 0.0, 0.042724,
     0.0000155488, 0.0000623689, 0.65392, 0.060167),
    (0.2026579, 0.331511, 0.027655, 0.203488,
     0.0313385, 0.0503618, 0.016901, 0.041577,
     0.000048736, 0.00000740336, 1.226, 0.03754),
)
REFERENCE_OMEGA = 0.3978
HYDROGEN = "1333-74-0"
NITROGEN = "7727-37-9"
CARBON_MONOXIDE = "630-08-0"
CARBON_DIOXIDE = "124-38-9"
HYDROGEN_SULFIDE = "7783-06-4"
DATA_PATH = Path(__file__).resolve().parents[1] / "data/lkp_binary_interactions_cas.json"


@lru_cache(maxsize=1)
def _tabulated_parameters():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {
        tuple(sorted((r["cas1"], r["cas2"]))): float(r["kij"])
        for r in payload["records"]
    }


def _hydrocarbon(formula: str) -> bool:
    # A formula, rather than a symbol/name, prevents mistaking CO/CH3OH
    # or arbitrary process aliases for hydrocarbons.
    return bool(re.fullmatch(r"(?:C\d*H\d*|H\d*C\d*)", formula or ""))


def estimate_kij(first, second) -> tuple[float, str]:
    """DWSIM missing-pair rules, applied to nominal resolved pure properties."""
    if second.CAS == HYDROGEN:
        first, second = second, first
    if first.CAS == HYDROGEN:
        value = (0.2036 + 0.002246 * second.Tc + 0.8364 * second.omega
                 + 1.2111 * (second.Vc / 1000.0)**(1.0 / 3.0))
        return value, "DWSIM hydrogen correlation"
    if second.CAS in {CARBON_DIOXIDE, HYDROGEN_SULFIDE}:
        first, second = second, first
    if (first.CAS in {CARBON_DIOXIDE, HYDROGEN_SULFIDE}
            and _hydrocarbon(second.formula) and second.Tc >= 304.21):
        X = math.log(second.Tc / 304.21) * math.log(second.Vc / 94.0)
        value = 0.9276 + 0.10719 * X + 0.04505 * X * X
        if first.CAS == HYDROGEN_SULFIDE:
            value -= 0.027
        return value, "DWSIM CO2/H2S-hydrocarbon correlation"
    if first.Tc > second.Tc:
        first, second = second, first
    if (_hydrocarbon(second.formula)
            and (_hydrocarbon(first.formula)
                 or first.CAS in {NITROGEN, CARBON_MONOXIDE})
            and second.Tc / first.Tc >= 1.3):
        ratio = math.log(second.Tc / first.Tc)
        value = 1.0 + 0.2802 * ratio * math.log(second.Vc / first.Vc) - 0.0561 * ratio
        return value, "DWSIM asymmetric hydrocarbon/light-gas correlation"
    return 1.0, "LKP neutral fallback (no tabulated or correlated parameter)"


def _coefficients(Tr, fluid):
    b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, beta, gamma = LK_CONSTANTS[fluid]
    return (b1 - b2 / Tr - b3 / Tr**2 - b4 / Tr**3,
            c1 - c2 / Tr + c3 / Tr**3, d1 + d2 / Tr, c4, beta, gamma)


def _pressure_and_slope(v, Tr, coefficients):
    """Reduced pressure and dPr/dVr; vectorized for spinodal bracketing."""
    B, C, D, c4, beta, gamma = coefficients
    u = gamma / v**2
    exponential = np.exp(-u)
    z = 1.0 + B / v + C / v**2 + D / v**5 + c4 / Tr**3 / v**2 * (beta + u) * exponential
    dz = (-B / v**2 - 2.0 * C / v**3 - 5.0 * D / v**6
          + c4 / Tr**3 / v**3 * exponential * (-2.0 * beta + (2.0 * beta - 4.0) * u + 2.0 * u**2))
    return Tr * z / v, Tr * (v * dz - z) / v**2


def _pressure_curvature(v, Tr, coefficients):
    B, C, D, c4, beta, gamma = coefficients
    u = gamma / v**2
    return Tr * (2.0 / v**3 + 6.0 * B / v**4 + 12.0 * C / v**5 + 42.0 * D / v**8
                 + c4 / Tr**3 / v**5 * np.exp(-u) * (
                     12.0 * beta + (30.0 - 18.0 * beta) * u
                     + (4.0 * beta - 26.0) * u**2 + 4.0 * u**3))


@lru_cache(maxsize=4096)
def _spinodals(Tr, fluid):
    coefficients = _coefficients(Tr, fluid)
    grid = np.geomspace(1e-3, 100.0, 320)
    # Bracket extrema of dP/dV first. A direct dP/dV sign grid can miss
    # both spinodals arbitrarily close to criticality, hiding liquid roots.
    curvatures = _pressure_curvature(grid, Tr, coefficients)
    indices = np.flatnonzero(curvatures[:-1] * curvatures[1:] < 0.0)
    extrema = [brentq(lambda v: _pressure_curvature(v, Tr, coefficients),
                     grid[i], grid[i + 1], xtol=1e-14) for i in indices]
    boundaries = (grid[0], *extrema, grid[-1])
    slope = lambda v: _pressure_and_slope(v, Tr, coefficients)[1]
    return tuple(brentq(slope, left, right, xtol=1e-14)
                 for left, right in zip(boundaries, boundaries[1:])
                 if slope(left) * slope(right) < 0.0)


@lru_cache(maxsize=16384)
def reduced_roots(Tr: float, Pr: float, fluid: int) -> tuple[float, ...]:
    """All mechanically stable reduced-volume roots, bracketed by spinodals.

Partitioning at extrema also finds roots close to saturation/criticality
that a pressure-residual grid can miss when two crossings nearly coincide.
"""
    if not (math.isfinite(Tr) and math.isfinite(Pr) and Tr > 0 and Pr > 0):
        raise ThermodynamicsError("LKP reduced temperature/pressure must be positive and finite")
    coefficients = _coefficients(Tr, fluid)
    low, high = 1e-3, max(100.0, 2.0 * Tr / Pr)
    residual = lambda v: _pressure_and_slope(v, Tr, coefficients)[0] - Pr
    while residual(low) < 0.0:
        low *= 0.5
    while residual(high) > 0.0:
        high *= 2.0
    boundaries = (low, *_spinodals(Tr, fluid), high)
    roots = []
    for left, right in zip(boundaries, boundaries[1:]):
        fl, fr = residual(left), residual(right)
        if fl * fr > 0.0:
            continue
        root = brentq(residual, left, right, xtol=1e-14, rtol=1e-14)
        # Classify the monotone interval, not its limiting root: roundoff in
        # dP/dV at a spinodal/critical root can have either sign.
        if _pressure_and_slope(math.sqrt(left * right), Tr, coefficients)[1] <= 0.0:
            if not roots or abs(root - roots[-1]) > 1e-12:
                roots.append(root)
    if not roots:
        raise ThermodynamicsError(f"LKP found no stable volume root at Tr={Tr:g}, Pr={Pr:g}")
    return tuple(roots)


def _fluid_state(Tr, Pr, fluid, phase):
    roots = reduced_roots(Tr, Pr, fluid)
    v = roots[0] if phase == "liquid" else roots[-1]
    z = Pr * v / Tr
    B, C, D, c4, beta, gamma = _coefficients(Tr, fluid)
    u = gamma / v**2
    # expm1 avoids loss of the small low-density departure term.
    E = c4 / (2.0 * Tr**3 * gamma) * ((beta + 1.0) * -math.expm1(-u) - u * math.exp(-u))
    g = z - 1.0 - math.log(z) + B / v + C / (2.0 * v**2) + D / (5.0 * v**5) + E
    _, b2, b3, b4, _, c2, c3, _, _, d2, _, _ = LK_CONSTANTS[fluid]
    h_RT = (z - 1.0 - (b2 + 2.0 * b3 / Tr + 3.0 * b4 / Tr**2) / (Tr * v)
            - (c2 - 3.0 * c3 / Tr**2) / (2.0 * Tr * v**2)
            + d2 / (5.0 * Tr * v**5) + 3.0 * E)
    return np.array((z, g, h_RT))


@dataclass(frozen=True)
class _Mixture:
    Tc: float
    Pc: float
    omega: float
    dTc_dx: np.ndarray
    dlogPc_dx: np.ndarray
    dTc_dT: float
    dlogPc_dT: float


class LKPThermodynamics(PhiPhiThermodynamics):
    """LKP corresponding-states EOS integrated with pfdsim's flash solvers."""

    model = "LKP"

    def __init__(self, components, db=None, interaction_overrides=None):
        super().__init__(components, db, interaction_overrides)
        if not components or len(set(components)) != len(components):
            raise ThermodynamicsError("LKP requires a nonempty list of unique components")
        self._index = {c: i for i, c in enumerate(self.components)}
        self._Tc = np.array([self.props[c].Tc for c in components], dtype=float)
        self._Vc = np.array([self.props[c].Vc for c in components], dtype=float) / 1000.0
        self._omega = np.array([self.props[c].omega for c in components], dtype=float)
        self._hydrogen_index = None
        for i, c in enumerate(components):
            p = self.props[c]
            if (not np.isfinite((self._Tc[i], self._Vc[i], self._omega[i])).all()
                    or self._Tc[i] <= 0 or self._Vc[i] <= 0
                    or p.Pc is None or not math.isfinite(p.Pc) or p.Pc <= 0):
                raise ThermodynamicsError(f"LKP requires finite Tc, Pc, Vc and omega, with positive criticals, for '{c}'")
            if p.CAS == HYDROGEN:
                self._hydrogen_index = i
                if not math.isfinite(p.MW) or p.MW <= 0:
                    raise ThermodynamicsError("LKP hydrogen quantum correction requires positive MW")
                self._omega[i] = 0.0
            self.mark_property_source_context(c, ("Tc", "Pc", "Vc", "omega"),
                                              kind="thermo_model", phase="lkp_parameters",
                                              description="LKP critical-property parameters", affects_result=True)
        self.kij = np.ones((len(components), len(components)))
        self.interaction_parameter_sources = {}
        table = _tabulated_parameters()
        overrides = {}
        for record in interaction_overrides or []:
            if str(record.get("model", "")).upper() != "LKP":
                continue
            pair = tuple(sorted((record["component1"], record["component2"])))
            if pair[0] == pair[1] or any(c not in self._index for c in pair):
                raise ThermodynamicsError(f"Invalid LKP interaction pair {pair}")
            if pair in overrides:
                raise ThermodynamicsError(f"Duplicate LKP interaction override for {pair}")
            value = record.get("kij", record.get("k_ij"))
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ThermodynamicsError("LKP kij must be positive and finite; neutral value is 1")
            overrides[pair] = float(value)
        for i, c1 in enumerate(components):
            for j in range(i + 1, len(components)):
                c2 = components[j]
                pair = tuple(sorted((c1, c2)))
                cas_pair = tuple(sorted((self.props[c1].CAS or "", self.props[c2].CAS or "")))
                if pair in overrides:
                    value, source = overrides[pair], "PFD interaction override"
                elif all(cas_pair) and cas_pair in table:
                    value, source = table[cas_pair], "DWSIM table (audited against Plocker 1978)"
                else:
                    value, source = estimate_kij(self.props[c1], self.props[c2])
                    self.add_warning(f"LKP {c1}/{c2}: kij={value:.6g}; {source}.")
                self.kij[i, j] = self.kij[j, i] = value
                self.interaction_parameter_sources[pair] = {"kij": value, "source": source}
        self._state_cache = {}

    def _normalized_composition(self, composition):
        try:
            return normalized_eos_composition(self.components, composition)
        except PhiPhiFlashError as error:
            raise ThermodynamicsError(f"LKP {error}") from error

    def _mixing(self, T, x):
        Tc, Vc = self._Tc.copy(), self._Vc.copy()
        dTc, dVc = np.zeros_like(Tc), np.zeros_like(Vc)
        i = self._hydrogen_index
        if i is not None:
            MW = self.props[self.components[i]].MW
            a, b = 21.8 / MW, 9.91 / MW
            if T <= b:
                raise ThermodynamicsError("LKP hydrogen quantum critical volume is invalid at this temperature")
            Tc[i] = 43.6 * T / (T + a)
            Vc[i] = 0.0515 * T / (T - b)
            dTc[i] = 43.6 * a / (T + a)**2
            dVc[i] = -0.0515 * b / (T - b)**2
        lengths = np.cbrt(Vc)
        pair_sum = lengths[:, None] + lengths[None, :]
        vpair = pair_sum**3 / 8.0
        tpair = np.sqrt(Tc[:, None] * Tc[None, :]) * self.kij
        apair = vpair**0.25 * tpair
        volume = float(x @ vpair @ x)
        temperature = float(x @ apair @ x) / volume**0.25
        omega = float(x @ self._omega)
        zc = 0.2905 - 0.085 * omega
        pressure = zc * (R / 100.0) * temperature / volume
        if zc <= 0 or not math.isfinite(pressure) or temperature <= 0:
            raise ThermodynamicsError("LKP mixing rule produced invalid pseudo-critical properties")
        dV_dx = 2.0 * vpair @ x
        dT_dx = 2.0 * apair @ x / volume**0.25 - 0.25 * temperature * dV_dx / volume
        dlogP_dx = dT_dx / temperature - dV_dx / volume - 0.085 * self._omega / zc
        dvpair = 3.0 / 8.0 * pair_sum**2 * (
            (dVc / (3.0 * lengths**2))[:, None] + (dVc / (3.0 * lengths**2))[None, :])
        dtpair = 0.5 * tpair * ((dTc / Tc)[:, None] + (dTc / Tc)[None, :])
        dapair = 0.25 * vpair**-0.75 * dvpair * tpair + vpair**0.25 * dtpair
        dV_dT = float(x @ dvpair @ x)
        dT_dT = float(x @ dapair @ x) / volume**0.25 - 0.25 * temperature * dV_dT / volume
        return _Mixture(temperature, pressure, omega, dT_dx, dlogP_dx,
                        dT_dT, dT_dT / temperature - dV_dT / volume)

    def _state(self, T, P, composition, phase):
        if not (math.isfinite(T) and math.isfinite(P) and T > 0 and P > 0):
            raise ThermodynamicsError("LKP temperature and pressure must be positive and finite")
        if phase not in {"vapor", "liquid"}:
            raise ThermodynamicsError("LKP phase must be 'vapor' or 'liquid'")
        normalized = self._normalized_composition(composition)
        x = np.array(list(normalized.values()))
        key = (float(T), float(P), tuple(x), phase)
        if key in self._state_cache:
            return self._state_cache[key]
        mixture = self._mixing(T, x)
        Tr, Pr = T / mixture.Tc, P / mixture.Pc
        simple = _fluid_state(Tr, Pr, 0, phase)
        reference = _fluid_state(Tr, Pr, 1, phase)
        z, g, h_RT = simple + mixture.omega / REFERENCE_OMEGA * (reference - simple)
        if not math.isfinite(z) or z <= 0:
            raise ThermodynamicsValidityError("LKP acentric interpolation produced nonpositive compressibility factor")
        # For fixed composition and T the pseudo-criticals are fixed. Hence
        # dVm/dP has the sign of the interpolated dVr/dPr. Stable roots of
        # the two fluids do not guarantee this sign when omega extrapolates.
        weight = mixture.omega / REFERENCE_OMEGA
        derivatives = []
        for fluid, state in enumerate((simple, reference)):
            volume = state[0] * Tr / Pr
            slope = float(_pressure_and_slope(volume, Tr, _coefficients(Tr, fluid))[1])
            # reduced_roots already selects stable monotone intervals. At a
            # limiting spinodal root cancellation may give a tiny positive
            # slope; use the stable-side infinite derivative there.
            derivatives.append(1.0 / slope if slope < 0.0 else -math.inf)
        dvolume_dpressure = ((1.0 - weight) * derivatives[0] + weight * derivatives[1]
                            if weight not in (0.0, 1.0) else derivatives[int(weight)])
        if math.isnan(dvolume_dpressure) or dvolume_dpressure >= 0.0:
            raise ThermodynamicsValidityError(
                "LKP acentric interpolation is mechanically unstable "
                f"(nonpositive isothermal compressibility) at T={T:g} K, "
                f"P={P:g} bar, phase={phase}, omega={mixture.omega:g}"
            )
        dg_dx = (h_RT * mixture.dTc_dx / mixture.Tc - (z - 1.0) * mixture.dlogPc_dx
                 + (reference[1] - simple[1]) / REFERENCE_OMEGA * self._omega)
        lnphi = g + dg_dx - float(x @ dg_dx)
        # Include temperature-dependent hydrogen pseudo-criticals in H=-RT²dg/dT.
        h = R * T * h_RT - R * T**2 * (
            h_RT * mixture.dTc_dT / mixture.Tc - (z - 1.0) * mixture.dlogPc_dT)
        state = (float(z), float(g), float(h), lnphi, mixture)
        return self._set_limited_cache(self._state_cache, key, state)

    def compressibility_factor(self, T, P, composition, phase="vapor"):
        return self._state(T, P, composition, phase)[0]

    def compressibility_roots(self, T, P, composition):
        liquid = self.compressibility_factor(T, P, composition, "liquid")
        vapor = self.compressibility_factor(T, P, composition, "vapor")
        return (liquid,) if abs(vapor - liquid) < 1e-12 else (liquid, vapor)

    def _eos_root_kind(self, T, P, composition, phase):
        mixture = self._state(T, P, composition, phase)[-1]
        Tr, Pr = T / mixture.Tc, P / mixture.Pc
        kinds = set()
        weight = mixture.omega / REFERENCE_OMEGA
        for fluid, coefficient in enumerate((1.0 - weight, weight)):
            if coefficient == 0.0:
                continue
            spinodals = _spinodals(Tr, fluid)
            if spinodals:
                roots = reduced_roots(Tr, Pr, fluid)
                volume = roots[0] if phase == 'liquid' else roots[-1]
                kinds.add('liquid' if volume <= spinodals[0] else 'vapor')
        if len(kinds) == 1:
            return kinds.pop()
        return 'indeterminate' if kinds else 'supercritical'

    def _eos_pressure_extrema(self, T, composition):
        x = np.array(list(self._normalized_composition(composition).values()))
        mixture = self._mixing(T, x)
        Tr = T / mixture.Tc
        weight = mixture.omega / REFERENCE_OMEGA
        return tuple(
            mixture.Pc * float(_pressure_and_slope(v, Tr, _coefficients(Tr, fluid))[0])
            for fluid, coefficient in enumerate((1-weight, weight)) if coefficient != 0
            for v in _spinodals(Tr, fluid)
        )

    def fugacity_coefficients(self, T, P, composition, phase="vapor"):
        lnphi = self._state(T, P, composition, phase)[3]
        if not np.isfinite(lnphi).all() or (lnphi > 700.0).any() or (lnphi < -700.0).any():
            raise ThermodynamicsError("LKP fugacity coefficient is outside the finite numerical range")
        return dict(zip(self.components, np.exp(lnphi).tolist()))

    def molar_volume(self, T, P, composition, phase="vapor"):
        return self.compressibility_factor(T, P, composition, phase) * (R / 100.0) * T / P

    def molar_volume_vapor(self, T, P, composition):
        return self.molar_volume(T, P, composition, "vapor")

    vapor_molar_volume_for_density = molar_volume_vapor

    def molar_volume_liquid(self, T, P, composition):
        return self.mixture_liquid_molar_volume(composition, T)

    def departure_gibbs(self, T, P, composition, phase="vapor"):
        return R * T * self._state(T, P, composition, phase)[1]

    def departure_enthalpy(self, T, P, composition, phase="vapor"):
        return self._state(T, P, composition, phase)[2]

    def departure_entropy(self, T, P, composition, phase="vapor"):
        state = self._state(T, P, composition, phase)
        return state[2] / T - R * state[1]

    def K_values(self, T, P, composition):
        # Import locally to avoid the public thermo facade during module loading.
        if __package__.split('.', 1)[0] == 'pfdsim':
            from ..phi_phi import iterate_phi_phi_K
        else:
            from phi_phi import iterate_phi_phi_K
        x = self._normalized_composition(composition)
        key = self._k_values_cache_key("lkp_phi_phi", T, P, x)
        cached = self._get_cached_k_values(key)
        if cached is not None:
            return cached
        return self._set_cached_k_values(key, iterate_phi_phi_K(self, T, P, x, 80))

    def aqueous_K_values(self, T, P, composition, context):
        return _phi_henry_aqueous_K_values(self, T, P, composition, context, "aqueous_lkp_phi_henry")
