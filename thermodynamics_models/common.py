"""
Thermodynamics Module - Ideal, activity-coefficient, and EOS property methods

Implements thermodynamic calculations for process simulation:

IDEAL Method:
- Ideal gas law: PV = nRT
- Raoult's law VLE: p_i = x_i * P_sat_i
- Ideal mixing (no excess properties)
- Heat capacity integration for enthalpy

Activity-coefficient methods:
- UNIFAC and NRTL liquid activity coefficients
- Gamma-phi variants with RK or PR vapor fugacity coefficients
- VLE, LLE, VLLE, and excess enthalpy utilities

EOS methods:
- RK, SRK, PR, RKS-BM, and PR-BM cubic equations of state
- Fugacity coefficients, molar volumes, and departure functions
"""

import math
import re
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalProperties
else:
    from chemical_properties import ChemicalProperties
from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..pressure_standards import THERMOCHEMICAL_STANDARD_PRESSURE_BAR
else:
    from pressure_standards import THERMOCHEMICAL_STANDARD_PRESSURE_BAR


# Constants
R = 8.314  # J/mol-K
R_BAR = 8.314e-5  # bar-m3/mol-K
T_REF = 298.15  # K (reference temperature for enthalpy)
P_REF = THERMOCHEMICAL_STANDARD_PRESSURE_BAR  # bar, thermochemical standard state
LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K = 20.0
DEFAULT_STATE_INCLUDE = frozenset({'H', 'Cp', 'S', 'rho'})
STEAM_WATER_MW = 18.0153  # kg/kmol, aligned with data/chemicals.json H2O
# CoolProp IF97 offsets calibrated so liquid water at 298.15 K and the
# 1 bar thermochemical standard has H=-285830 kJ/kmol and S=69.95 kJ/kmol-K.


def _looks_like_molecular_formula(value: Optional[str]) -> bool:
    if not value:
        return False
    text = str(value).strip()
    return bool(
        re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", text)
        and any(ch.isdigit() for ch in text)
    )


def _property_lookup_identifier(comp: str, props: ChemicalProperties) -> str:
    """Prefer specific identity fields over ambiguous formula-like symbols."""
    if props.symbol and (
        not _looks_like_molecular_formula(props.symbol)
        or (
            props.formula
            and str(props.symbol).strip().lower() != str(props.formula).strip().lower()
        )
    ):
        return props.symbol
    for candidate in (props.CAS, props.name, comp, props.symbol):
        if candidate:
            return str(candidate)
    return comp
STEAM_WATER_H_OFFSET = -287720.31061464705  # kJ/kmol
STEAM_WATER_S_OFFSET = 63.33421692723281  # kJ/kmol-K


class ThermodynamicsError(Exception):
    """Error in thermodynamic calculations"""
    pass


def _solve_bubble_point_temperature(thermo, composition: dict[str, float],
                                    P: float, T_guess: float = 350.0) -> float:
    """Solve sum(x_i K_i) = 1 with bracketing around available pure data."""
    x_total = sum(max(float(value), 0.0) for value in composition.values())
    if x_total <= 0.0:
        raise ThermodynamicsError("Cannot calculate bubble point for zero composition")
    x = {
        comp: max(float(value), 0.0) / x_total
        for comp, value in composition.items()
    }

    def residual(T: float) -> float:
        K = thermo.K_values(T, P, x)
        return sum(x.get(comp, 0.0) * K.get(comp, 1.0) for comp in x) - 1.0

    significant_cutoff = 1e-6
    significant_components = [
        comp for comp, value in x.items()
        if value >= significant_cutoff
    ]
    if not significant_components:
        significant_components = [max(x, key=x.get)]

    def pure_saturation_temperature(comp: str) -> Optional[float]:
        """Estimate pure-component saturation temperature at P on a broad scan."""
        saturation_cache = getattr(thermo, "_pure_saturation_temperature_cache", None)
        cache_key = (comp, round(float(P), 10))
        if isinstance(saturation_cache, dict) and cache_key in saturation_cache:
            return saturation_cache[cache_key]

        props = getattr(thermo, "props", {}).get(comp)
        target_pressure = P
        pressure_capped = False
        if props and props.Pc and P >= float(props.Pc):
            target_pressure = 0.999 * float(props.Pc)
            pressure_capped = True
            if hasattr(thermo, "add_warning"):
                thermo.add_warning(
                    f"Bubble-point bracketing pressure {P:.4g} bar is at or above "
                    f"the pure critical pressure for {comp}; using 0.999*Pc as "
                    "the pure-component saturation target for bracketing only."
                )
        broad_candidates = [
            80.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350.0,
            400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0,
            float(T_guess),
        ]
        if props:
            for value in (props.Tb, props.Tc):
                if value:
                    broad_candidates.extend([0.55 * value, value, 0.8 * value, 1.25 * value])
            if props.Tc:
                broad_candidates.append(0.999 * float(props.Tc))
        upper_bound = 1500.0
        if props and props.Tc:
            upper_bound = min(upper_bound, 0.999 * float(props.Tc))
        scan_grid = sorted({
            max(1.0, min(upper_bound, float(T)))
            for T in broad_candidates
        })
        sat_values = []
        for T in scan_grid:
            try:
                sat_values.append((T, thermo.Psat(comp, T) - target_pressure))
            except Exception:
                continue
        sat_roots = []
        for (T1, f1), (T2, f2) in zip(sat_values, sat_values[1:]):
            if abs(f1) < 1e-8:
                sat_roots.append(T1)
            elif f1 * f2 < 0:
                try:
                    sat_roots.append(
                        brentq(
                            lambda T: thermo.Psat(comp, T) - target_pressure,
                            T1,
                            T2,
                            xtol=1e-7,
                            rtol=1e-9,
                            maxiter=100,
                        )
                    )
                except Exception:
                    continue
        if not sat_roots:
            if pressure_capped and props and props.Tc:
                result = 0.999 * float(props.Tc)
                if isinstance(saturation_cache, dict):
                    saturation_cache[cache_key] = result
                return result
            return None
        Tb = float(props.Tb) if props and props.Tb else float(T_guess)
        result = min(sat_roots, key=lambda root: abs(root - Tb))
        if isinstance(saturation_cache, dict):
            if len(saturation_cache) > 20000:
                saturation_cache.clear()
            saturation_cache[cache_key] = result
        return result

    pure_targets = {}
    boiling_points = []
    for comp in significant_components:
        props = getattr(thermo, "props", {}).get(comp)
        target = pure_saturation_temperature(comp)
        if target is not None:
            pure_targets[comp] = target
        elif props and props.Tb:
            boiling_points.append(float(props.Tb))

    bracket_basis = list(pure_targets.values()) or boiling_points

    if bracket_basis:
        T_low = max(1.0, min(bracket_basis) - 60.0)
        T_high = min(1500.0, max(bracket_basis) + 100.0)
        candidate_temperatures = [
            T_low + index * (T_high - T_low) / 40.0
            for index in range(41)
        ]
        candidate_temperatures.extend(bracket_basis)
        candidate_temperatures.append(max(T_low, min(T_high, float(T_guess))))
    else:
        candidate_temperatures = [
            T_guess, 150.0, 200.0, 250.0, 300.0, 350.0,
            400.0, 500.0, 650.0, 800.0, 1000.0,
        ]
        for comp in x:
            props = getattr(thermo, "props", {}).get(comp)
            if props:
                for value in (props.Tb, props.Tc):
                    if value:
                        candidate_temperatures.extend([0.55 * value, value, 1.25 * value])

    grid = sorted(set(max(1.0, min(1500.0, float(T))) for T in candidate_temperatures))

    target_temperature = None
    target_weight = 0.0
    target_sum = 0.0
    for comp in significant_components:
        props = getattr(thermo, "props", {}).get(comp)
        target = pure_targets.get(comp)
        if target is None and props and props.Tb:
            target = float(props.Tb)
        if target is None:
            continue
        weight = x.get(comp, 0.0)
        target_sum += weight * target
        target_weight += weight
    if target_weight > 0.0:
        target_temperature = target_sum / target_weight

    if target_temperature is not None:
        local_centers = [target_temperature]
        T_guess_clipped = max(1.0, min(1500.0, float(T_guess)))
        if abs(T_guess_clipped - target_temperature) > 1e-9:
            local_centers.append(T_guess_clipped)
        for center_index, center in enumerate(local_centers):
            widths = (8.0, 16.0, 32.0, 64.0) if center_index == 0 else (16.0, 32.0)
            for width in widths:
                T1 = max(1.0, center - width)
                T2 = min(1500.0, center + width)
                if T2 <= T1:
                    continue
                try:
                    f1 = residual(T1)
                    if abs(f1) < 1e-8:
                        return T1
                    f2 = residual(T2)
                    if abs(f2) < 1e-8:
                        return T2
                    if f1 * f2 < 0:
                        return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
                except Exception:
                    continue

    values = []
    for T in grid:
        try:
            values.append((T, residual(T)))
        except Exception:
            continue
    roots = []
    for (T1, f1), (T2, f2) in zip(values, values[1:]):
        if abs(f1) < 1e-8:
            roots.append(T1)
        if f1 * f2 < 0:
            try:
                roots.append(brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100))
            except Exception:
                continue

    if roots:
        unique_roots = sorted({round(root, 8): root for root in roots}.values())
        if target_temperature is None:
            target_temperature = max(grid[0], min(grid[-1], float(T_guess)))
        return min(
            unique_roots,
            key=lambda root: (
                abs(root - target_temperature),
                abs(root - max(grid[0], min(grid[-1], float(T_guess)))),
            ),
        )

    if values:
        best_temperature, best_residual = min(
            values,
            key=lambda item: abs(item[1]),
        )
        if abs(best_residual) <= 1.0e-8:
            return best_temperature
        raise ThermodynamicsError(
            "Could not bracket bubble point temperature; "
            f"best residual {best_residual:.6g} at "
            f"T={best_temperature:.6g} K"
        )
    raise ThermodynamicsError(
        "Could not bracket bubble point temperature; no valid residual "
        "evaluations were available"
    )


def _solve_dew_point_temperature(thermo, composition: dict[str, float],
                                 P: float, T_guess: float = 350.0) -> float:
    """Solve sum(y_i / K_i) = 1 while iterating incipient liquid composition."""
    components = list(getattr(thermo, "components", composition.keys()))
    y_raw = {
        comp: max(float(composition.get(comp, 0.0)), 0.0)
        for comp in components
    }
    y_total = sum(y_raw.values())
    if y_total <= 0.0:
        raise ThermodynamicsError("Cannot calculate dew point for zero composition")
    y = {comp: value / y_total for comp, value in y_raw.items()}

    def residual(T: float) -> float:
        x = dict(y)
        sum_yK_inv = 1.0
        for _ in range(50):
            K = thermo.K_values(T, P, x)
            x_new = {}
            sum_yK_inv = 0.0
            for comp in components:
                K_i = max(1e-12, K.get(comp, 1.0))
                value = y.get(comp, 0.0) / K_i
                x_new[comp] = value
                sum_yK_inv += value
            if sum_yK_inv <= 0.0:
                break
            x_new = {comp: value / sum_yK_inv for comp, value in x_new.items()}
            max_change = max(
                abs(x_new.get(comp, 0.0) - x.get(comp, 0.0))
                for comp in components
            )
            x = x_new
            if max_change < 1e-9:
                break
        return sum_yK_inv - 1.0

    significant_cutoff = 1e-6
    significant_components = [
        comp for comp, value in y.items()
        if value >= significant_cutoff
    ]
    if not significant_components:
        significant_components = [max(y, key=y.get)]

    def pure_saturation_temperature(comp: str) -> Optional[float]:
        saturation_cache = getattr(thermo, "_pure_saturation_temperature_cache", None)
        cache_key = (comp, round(float(P), 10))
        if isinstance(saturation_cache, dict) and cache_key in saturation_cache:
            return saturation_cache[cache_key]

        props = getattr(thermo, "props", {}).get(comp)
        target_pressure = P
        pressure_capped = False
        if props and props.Pc and P >= float(props.Pc):
            target_pressure = 0.999 * float(props.Pc)
            pressure_capped = True
            if hasattr(thermo, "add_warning"):
                thermo.add_warning(
                    f"Dew-point bracketing pressure {P:.4g} bar is at or above "
                    f"the pure critical pressure for {comp}; using 0.999*Pc as "
                    "the pure-component saturation target for bracketing only."
                )

        broad_candidates = [
            80.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350.0,
            400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0,
            float(T_guess),
        ]
        if props:
            for value in (props.Tb, props.Tc):
                if value:
                    broad_candidates.extend([0.55 * value, value, 0.8 * value, 1.25 * value])
            if props.Tc:
                broad_candidates.append(0.999 * float(props.Tc))

        upper_bound = 1500.0
        if props and props.Tc:
            upper_bound = min(upper_bound, 0.999 * float(props.Tc))
        scan_grid = sorted({
            max(1.0, min(upper_bound, float(T)))
            for T in broad_candidates
        })
        sat_values = []
        for T in scan_grid:
            try:
                sat_values.append((T, thermo.Psat(comp, T) - target_pressure))
            except Exception:
                continue

        sat_roots = []
        for (T1, f1), (T2, f2) in zip(sat_values, sat_values[1:]):
            if abs(f1) < 1e-8:
                sat_roots.append(T1)
            elif f1 * f2 < 0:
                try:
                    sat_roots.append(
                        brentq(
                            lambda T: thermo.Psat(comp, T) - target_pressure,
                            T1,
                            T2,
                            xtol=1e-7,
                            rtol=1e-9,
                            maxiter=100,
                        )
                    )
                except Exception:
                    continue
        if not sat_roots:
            if pressure_capped and props and props.Tc:
                result = 0.999 * float(props.Tc)
                if isinstance(saturation_cache, dict):
                    saturation_cache[cache_key] = result
                return result
            return None

        Tb = float(props.Tb) if props and props.Tb else float(T_guess)
        result = min(sat_roots, key=lambda root: abs(root - Tb))
        if isinstance(saturation_cache, dict):
            if len(saturation_cache) > 20000:
                saturation_cache.clear()
            saturation_cache[cache_key] = result
        return result

    pure_targets = {}
    boiling_points = []
    for comp in significant_components:
        props = getattr(thermo, "props", {}).get(comp)
        target = pure_saturation_temperature(comp)
        if target is not None:
            pure_targets[comp] = target
        elif props and props.Tb:
            boiling_points.append(float(props.Tb))

    target_temperature = None
    target_weight = 0.0
    target_sum = 0.0
    for comp in significant_components:
        props = getattr(thermo, "props", {}).get(comp)
        target = pure_targets.get(comp)
        if target is None and props and props.Tb:
            target = float(props.Tb)
        if target is None:
            continue
        weight = y.get(comp, 0.0)
        target_sum += weight * target
        target_weight += weight
    if target_weight > 0.0:
        target_temperature = target_sum / target_weight

    T_guess_clipped = max(1.0, min(1500.0, float(T_guess)))
    local_centers = [T_guess_clipped]
    if target_temperature is not None:
        local_centers.append(target_temperature)
    for center_index, center in enumerate(local_centers):
        if center_index > 0 and any(abs(center - prior) <= 1e-9 for prior in local_centers[:center_index]):
            continue
        widths = (4.0, 8.0, 16.0, 32.0, 64.0) if center_index == 0 else (4.0, 8.0, 16.0, 32.0)
        try:
            f_center = residual(center)
            if math.isfinite(f_center) and abs(f_center) < 1e-8:
                return center
        except Exception:
            pass
        for width in widths:
            T1 = max(1.0, center - width)
            T2 = min(1500.0, center + width)
            if T2 <= T1:
                continue
            try:
                f1 = residual(T1)
                if abs(f1) < 1e-8:
                    return T1
                f2 = residual(T2)
                if abs(f2) < 1e-8:
                    return T2
                if f1 * f2 < 0:
                    return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
            except Exception:
                continue

    candidates = [
        80.0, 120.0, 150.0, 200.0, 250.0, 273.15, 298.15, 320.0,
        350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0,
        float(T_guess),
        float(T_guess) - 120.0, float(T_guess) - 60.0, float(T_guess) - 20.0,
        float(T_guess) + 20.0, float(T_guess) + 60.0, float(T_guess) + 120.0,
    ]
    if pure_targets:
        targets = list(pure_targets.values())
        T_low = max(1.0, min(targets) - 60.0)
        T_high = min(1500.0, max(targets) + 100.0)
        candidates.extend(T_low + index * (T_high - T_low) / 40.0 for index in range(41))
        candidates.extend(targets)
    elif boiling_points:
        T_low = max(1.0, min(boiling_points) - 60.0)
        T_high = min(1500.0, max(boiling_points) + 100.0)
        candidates.extend(T_low + index * (T_high - T_low) / 40.0 for index in range(41))
        candidates.extend(boiling_points)
    for comp in components:
        props = getattr(thermo, "props", {}).get(comp)
        if props:
            for value in (props.Tb, props.Tc):
                if value:
                    candidates.extend([0.55 * value, value, 0.8 * value, 1.25 * value])

    grid = sorted(set(max(1.0, min(1500.0, float(T))) for T in candidates))
    values = []
    for T in grid:
        try:
            value = residual(T)
            if math.isfinite(value):
                values.append((T, value))
        except Exception:
            continue

    for T, value in values:
        if abs(value) < 1e-8:
            return T

    roots = []
    for (T1, f1), (T2, f2) in zip(values, values[1:]):
        if f1 * f2 < 0:
            try:
                roots.append(brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100))
            except Exception:
                continue

    if roots:
        guess = max(grid[0], min(grid[-1], float(T_guess)))
        return min(roots, key=lambda root: abs(root - guess))
    if values:
        return min(values, key=lambda item: abs(item[1]))[0]
    raise ThermodynamicsError("Could not bracket dew point temperature")

__all__ = [
    'R', 'R_BAR', 'T_REF', 'P_REF',
    'LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K', 'DEFAULT_STATE_INCLUDE',
    'STEAM_WATER_MW', 'STEAM_WATER_H_OFFSET', 'STEAM_WATER_S_OFFSET',
    'ThermodynamicsError',
]
