"""
Redlich-Kwong Equation of State Module

Implements the Redlich-Kwong (RK) equation of state for vapor-phase properties
and VLE calculations.

RK Equation:
    P = RT/(V-b) - a/(T^0.5 * V * (V+b))

where:
    a = 0.42748 * R^2 * Tc^2.5 / Pc
    b = 0.08664 * R * Tc / Pc

Mixing rules (van der Waals one-fluid):
    a_mix = sum_i sum_j (y_i * y_j * sqrt(a_i * a_j))
    b_mix = sum_i (y_i * b_i)

References:
    Redlich, O.; Kwong, J.N.S. (1949). Chem. Rev. 44: 233-244.
"""

import math
from typing import Optional
from dataclasses import dataclass
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_BAR_CM3_MOL_K, R_BAR_M3_MOL_K, R_J_MOL_K
else:
    from physical_constants import R_BAR_CM3_MOL_K, R_BAR_M3_MOL_K, R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .chemical_properties import ChemicalProperties, ChemicalDatabase, get_database
else:
    from chemical_properties import ChemicalProperties, ChemicalDatabase, get_database


# Gas constant in different units
R = R_J_MOL_K  # J/mol-K
R_BAR = R_BAR_M3_MOL_K  # bar-m3/mol-K (converted from SI)
R_CM3 = R_BAR_CM3_MOL_K  # bar-cm3/mol-K


class RKError(Exception):
    """Error in RK calculations"""
    pass


@dataclass
class RKParams:
    """RK parameters for a pure component"""
    a: float  # bar-cm6-K0.5/mol2
    b: float  # cm3/mol
    Tc: float  # K
    Pc: float  # bar
    
    @classmethod
    def from_critical(cls, Tc: float, Pc: float) -> 'RKParams':
        """Create RK parameters from critical properties"""
        # a in bar-cm6-K0.5/mol2
        a = 0.42748 * R_CM3**2 * Tc**2.5 / Pc
        # b in cm3/mol
        b = 0.08664 * R_CM3 * Tc / Pc
        return cls(a=a, b=b, Tc=Tc, Pc=Pc)


class RedlichKwong:
    """
    Redlich-Kwong equation of state calculator.
    
    Usage:
        rk = RedlichKwong(['CH4', 'C2H6', 'C3H8'])
        Z = rk.compressibility_factor(T, P, composition)
        phi = rk.fugacity_coefficients(T, P, composition)
    """
    
    def __init__(self, components: list[str], db: Optional[ChemicalDatabase] = None):
        """
        Initialize RK calculator with component list.
        
        Args:
            components: List of component symbols
            db: Chemical database (uses default if None)
        """
        self.db = db or get_database()
        self.components = components
        self.props: dict[str, ChemicalProperties] = {}
        self.params: dict[str, RKParams] = {}
        self._fugacity_cache: dict[tuple, dict[str, float]] = {}
        self._phi_phi_k_cache: dict[tuple, dict[str, float]] = {}
        self.warnings: list[str] = []
        
        # Load properties and calculate RK parameters
        for comp in components:
            props = self.db.get(comp)
            if props is None:
                raise RKError(f"Component '{comp}' not found in database")
            
            self.props[comp] = props
            
            # Need critical properties for RK
            if props.Tc is None or props.Pc is None:
                raise RKError(
                    f"Component '{comp}' missing critical properties (Tc, Pc) "
                    "required for RK equation of state"
                )
            
            self.params[comp] = RKParams.from_critical(props.Tc, props.Pc)

        if len(components) > 1:
            self.warnings.append(
                "RK EOS binary interaction parameters are not implemented; "
                "using classical van der Waals mixing with k_ij=0 for all component pairs."
            )
        self._zero_kij = tuple(0.0 for _ in range(len(components) ** 2))
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compiled_cubic_eos import CompiledCubicEOSBackend
            else:
                from compiled_cubic_eos import CompiledCubicEOSBackend

            self._compiled_backend = CompiledCubicEOSBackend.from_rk(self)
        except Exception:
            self._compiled_backend = None

    def _composition_cache_key(self, composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(
            (comp, float(value))
            for comp, value in sorted(composition.items())
        )

    def _normalized_composition(self, composition: dict[str, float]) -> dict[str, float]:
        values = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        total = sum(values.values())
        if total <= 0.0:
            return {comp: 1.0 / len(self.components) for comp in self.components}
        return {comp: value / total for comp, value in values.items()}
    
    def mixture_params(self, composition: dict[str, float]) -> tuple[float, float]:
        """
        Calculate mixture a and b using van der Waals mixing rules.
        
        Args:
            composition: Mole fractions
            
        Returns:
            (a_mix, b_mix)
        """
        # a_mix = sum_i sum_j (y_i * y_j * sqrt(a_i * a_j))
        a_mix = 0.0
        for comp_i, y_i in composition.items():
            if comp_i not in self.params:
                continue
            a_i = self.params[comp_i].a
            for comp_j, y_j in composition.items():
                if comp_j not in self.params:
                    continue
                a_j = self.params[comp_j].a
                a_mix += y_i * y_j * math.sqrt(a_i * a_j)
        
        # b_mix = sum_i (y_i * b_i)
        b_mix = sum(
            y * self.params[comp].b 
            for comp, y in composition.items() 
            if comp in self.params
        )
        
        return a_mix, b_mix
    
    def compressibility_cubic(self, T: float, P: float, 
                               composition: dict[str, float]) -> list[float]:
        """
        Solve cubic equation for compressibility factor Z.
        
        RK in terms of Z:
            Z^3 - Z^2 + (A - B - B^2)*Z - A*B = 0
            
        where:
            A = a*P / (R^2 * T^2.5)
            B = b*P / (R*T)
            
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            
        Returns:
            List of real positive roots (usually 1 or 3)
        """
        if self._compiled_backend is not None:
            values = [
                float(composition.get(component, 0.0))
                for component in self.components
            ]
            return [
                float(value)
                for value in self._compiled_backend.compressibility_roots(
                    T, P, values, self._zero_kij
                )
            ]

        a_mix, b_mix = self.mixture_params(composition)
        
        # Dimensionless parameters
        A = a_mix * P / (R_CM3**2 * T**2.5)
        B = b_mix * P / (R_CM3 * T)
        
        # Cubic coefficients: Z^3 + c2*Z^2 + c1*Z + c0 = 0
        c2 = -1.0
        c1 = A - B - B**2
        c0 = -A * B
        
        # Solve cubic using Cardano's formula
        roots = self._solve_cubic(1.0, c2, c1, c0)
        
        # Filter for positive real roots
        valid_roots = [z for z in roots if z > 0]
        
        return sorted(valid_roots)
    
    def _solve_cubic(self, a: float, b: float, c: float, d: float) -> list[float]:
        """
        Solve cubic equation ax^3 + bx^2 + cx + d = 0
        Returns list of real roots.
        """
        # Normalize
        b /= a
        c /= a
        d /= a
        
        # Depressed cubic: t^3 + p*t + q = 0 where x = t - b/3
        p = c - b**2 / 3
        q = 2*b**3/27 - b*c/3 + d
        
        # Discriminant
        disc = (q/2)**2 + (p/3)**3
        
        roots = []
        
        if disc > 1e-10:
            # One real root
            sqrt_disc = math.sqrt(disc)
            u = (-q/2 + sqrt_disc)
            v = (-q/2 - sqrt_disc)
            u = math.copysign(abs(u)**(1/3), u)
            v = math.copysign(abs(v)**(1/3), v)
            t = u + v
            roots.append(t - b/3)
        else:
            # Three real roots
            if abs(p) < 1e-10:
                roots = [-b/3, -b/3, -b/3]
            else:
                r = math.sqrt(-p**3 / 27)
                theta = math.acos(max(-1, min(1, -q / (2*r)))) / 3
                m = 2 * (-p/3)**0.5
                
                roots.append(m * math.cos(theta) - b/3)
                roots.append(m * math.cos(theta + 2*math.pi/3) - b/3)
                roots.append(m * math.cos(theta + 4*math.pi/3) - b/3)
        
        return roots
    
    def compressibility_factor(self, T: float, P: float, 
                                composition: dict[str, float],
                                phase: str = 'vapor') -> float:
        """
        Get compressibility factor Z for specified phase.
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Z value
        """
        roots = self.compressibility_cubic(T, P, composition)
        
        if not roots:
            # Fall back to ideal gas
            return 1.0
        
        if phase == 'vapor':
            return max(roots)  # Largest root for vapor
        else:
            return min(roots)  # Smallest root for liquid
    
    def molar_volume(self, T: float, P: float, composition: dict[str, float],
                     phase: str = 'vapor') -> float:
        """
        Calculate molar volume [cm3/mol]
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Molar volume [cm3/mol]
        """
        Z = self.compressibility_factor(T, P, composition, phase)
        return Z * R_CM3 * T / P
    
    def fugacity_coefficients(self, T: float, P: float,
                               composition: dict[str, float],
                               phase: str = 'vapor') -> dict[str, float]:
        """
        Calculate fugacity coefficients for each component.
        
        ln(phi_i) = (b_i/b_mix)*(Z-1) - ln(Z-B) 
                    - (A/B)*(2*sum_j(y_j*sqrt(a_i*a_j))/a_mix - b_i/b_mix)*ln(1+B/Z)
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Dictionary of fugacity coefficients
        """
        cache_key = (float(T), float(P), phase, self._composition_cache_key(composition))
        cached = self._fugacity_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if self._compiled_backend is not None:
            values = [
                float(composition.get(component, 0.0))
                for component in self.components
            ]
            compiled = self._compiled_backend.fugacity_coefficients(
                T, P, values, phase, self._zero_kij
            )
            by_component = {
                component: float(value)
                for component, value in zip(self.components, compiled)
            }
            phi = {
                component: by_component.get(component, 1.0)
                for component in composition
            }
            if len(self._fugacity_cache) > 20000:
                self._fugacity_cache.clear()
            self._fugacity_cache[cache_key] = dict(phi)
            return phi

        a_mix, b_mix = self.mixture_params(composition)
        Z = self.compressibility_factor(T, P, composition, phase)
        
        A = a_mix * P / (R_CM3**2 * T**2.5)
        B = b_mix * P / (R_CM3 * T)
        
        phi = {}
        
        for comp in composition:
            if comp not in self.params:
                phi[comp] = 1.0
                continue
            
            a_i = self.params[comp].a
            b_i = self.params[comp].b
            
            # Calculate cross term sum
            sum_aij = 0.0
            for comp_j, y_j in composition.items():
                if comp_j in self.params:
                    a_j = self.params[comp_j].a
                    sum_aij += y_j * math.sqrt(a_i * a_j)
            
            # Fugacity coefficient
            if Z > B and B > 0:
                ln_phi = (b_i / b_mix) * (Z - 1) - math.log(Z - B)
                ln_phi -= (A / B) * (2 * sum_aij / a_mix - b_i / b_mix) * math.log(1 + B / Z)
                phi[comp] = math.exp(ln_phi)
            else:
                phi[comp] = 1.0
        
        if len(self._fugacity_cache) > 20000:
            self._fugacity_cache.clear()
        self._fugacity_cache[cache_key] = dict(phi)
        return phi
    
    def departure_enthalpy(self, T: float, P: float,
                           composition: dict[str, float],
                           phase: str = 'vapor') -> float:
        """
        Calculate departure enthalpy (H - H_ig) [J/mol]
        
        (H - H_ig) = RT*(Z-1) - (3a)/(2bT^0.5) * ln(1 + bP/(ZRT))
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Departure enthalpy [J/mol]
        """
        if self._compiled_backend is not None:
            values = [
                float(composition.get(component, 0.0))
                for component in self.components
            ]
            return self._compiled_backend.departure_enthalpy(
                T, P, values, phase, self._zero_kij, self._zero_kij
            )
        a_mix, b_mix = self.mixture_params(composition)
        Z = self.compressibility_factor(T, P, composition, phase)
        
        B = b_mix * P / (R_CM3 * T)
        
        # Convert a to consistent units (J-cm3-K0.5/mol2)
        a_J = a_mix * 0.1  # bar-cm3 to J (1 bar-cm3 = 0.1 J)
        
        H_dep = R * T * (Z - 1)
        if b_mix > 0 and Z > B:
            H_dep -= (1.5 * a_J) / (b_mix * math.sqrt(T)) * math.log(1 + B / Z)
        
        return H_dep
    
    def departure_entropy(self, T: float, P: float,
                          composition: dict[str, float],
                          phase: str = 'vapor') -> float:
        """
        Calculate departure entropy (S - S_ig) [J/mol-K]
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Departure entropy [J/mol-K]
        """
        if self._compiled_backend is not None:
            values = [
                float(composition.get(component, 0.0))
                for component in self.components
            ]
            return self._compiled_backend.departure_entropy(
                T, P, values, phase, self._zero_kij, self._zero_kij
            )
        a_mix, b_mix = self.mixture_params(composition)
        Z = self.compressibility_factor(T, P, composition, phase)
        
        B = b_mix * P / (R_CM3 * T)
        
        a_J = a_mix * 0.1  # Convert to J units
        
        S_dep = R * math.log(Z - B) if Z > B else 0.0
        if b_mix > 0 and Z > B:
            S_dep -= (0.5 * a_J) / (b_mix * T**1.5) * math.log(1 + B / Z)
        
        return S_dep
    
    def K_values(self, T: float, P: float, x: dict[str, float],
                 y: dict[str, float]) -> dict[str, float]:
        """
        Calculate K-values using fugacity coefficients.
        
        K_i = phi_i^L / phi_i^V
        
        At VLE: y_i * phi_i^V = x_i * phi_i^L
        So: K_i = y_i / x_i = phi_i^L / phi_i^V
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            x: Liquid mole fractions
            y: Vapor mole fractions
            
        Returns:
            K-values for each component
        """
        phi_L = self.fugacity_coefficients(T, P, x, 'liquid')
        phi_V = self.fugacity_coefficients(T, P, y, 'vapor')
        
        K = {}
        for comp in self.components:
            phi_l = phi_L.get(comp, 1.0)
            phi_v = phi_V.get(comp, 1.0)
            if phi_v > 0:
                K[comp] = phi_l / phi_v
            else:
                K[comp] = 1.0
        
        return K

    def phi_phi_K_values(
        self,
        T: float,
        P: float,
        liquid_composition: dict[str, float],
        max_iter: int = 80,
    ) -> dict[str, float]:
        """Calculate VLE K-values from RK liquid/vapor fugacity coefficients."""
        x = self._normalized_composition(liquid_composition)
        cache_key = (
            float(T),
            float(P),
            self._composition_cache_key(x),
            int(max_iter),
        )
        cached = self._phi_phi_k_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if self._compiled_backend is not None:
            values = self._compiled_backend.phi_phi_K_values(
                T,
                P,
                [x[component] for component in self.components],
                max_iter,
                self._zero_kij,
            )
            result = {
                component: float(value)
                for component, value in zip(self.components, values)
            }
            return self._set_cached_phi_phi_K(cache_key, result)

        single_root = len(self.compressibility_cubic(T, P, x)) < 2

        phi_l = self.fugacity_coefficients(T, P, x, 'liquid')
        K = self._wilson_K(T, P)
        initial_K = dict(K)
        y = self._normalized_composition({
            comp: x[comp] * K[comp]
            for comp in self.components
        })

        for _ in range(max_iter):
            phi_v = self.fugacity_coefficients(T, P, y, 'vapor')
            K_new = {
                comp: max(
                    1e-8,
                    min(1e8, phi_l.get(comp, 1.0) / max(phi_v.get(comp, 1.0), 1e-12)),
                )
                for comp in self.components
            }
            y_new = self._normalized_composition({
                comp: x[comp] * K_new[comp]
                for comp in self.components
            })
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in self.components) < 1e-9:
                if single_root and all(abs(K_new[c] - 1.0) < 1e-6 for c in self.components if x[c] > 0.0):
                    return self._set_cached_phi_phi_K(cache_key, initial_K)
                return self._set_cached_phi_phi_K(cache_key, K_new)
            K = {
                comp: 0.5 * K[comp] + 0.5 * K_new[comp]
                for comp in self.components
            }
            y = y_new
        return self._set_cached_phi_phi_K(cache_key, K)

    def _set_cached_phi_phi_K(self, key: tuple, values: dict[str, float]) -> dict[str, float]:
        if len(self._phi_phi_k_cache) > 20000:
            self._phi_phi_k_cache.clear()
        self._phi_phi_k_cache[key] = dict(values)
        return values

    def _wilson_K(self, T: float, P: float) -> dict[str, float]:
        values = {}
        for comp in self.components:
            props = self.props[comp]
            omega = props.omega if props.omega is not None else 0.0
            exponent = 5.373 * (1.0 + omega) * (1.0 - props.Tc / T)
            values[comp] = max(1e-8, min(1e8, props.Pc / max(P, 1e-12) * math.exp(exponent)))
        return values
    
    def K_values_simplified(self, T: float, P: float) -> dict[str, float]:
        """
        Simplified K-values using Psat and vapor fugacity coefficient.
        
        K_i = Psat_i / (phi_i^V * P)
        
        This assumes liquid phase is ideal (phi^L ≈ 1).
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            
        Returns:
            K-values for each component
        """
        K = {}
        
        for comp in self.components:
            props = self.props[comp]
            
            # Get vapor pressure
            Psat = props.Psat(T) if hasattr(props, 'Psat') else None
            
            if Psat is None or props.Tc is None:
                # Above critical or no Psat - use high K for light components
                if props.Tc and T > props.Tc:
                    K[comp] = 10.0  # Supercritical - prefers vapor
                else:
                    K[comp] = 1.0
                continue
            
            # Calculate vapor fugacity coefficient at pure component
            pure_comp = {comp: 1.0}
            phi_v = self.fugacity_coefficients(T, P, pure_comp, 'vapor').get(comp, 1.0)
            
            # K = Psat / (phi_v * P)
            if phi_v > 0 and P > 0:
                K[comp] = Psat / (phi_v * P)
            else:
                K[comp] = Psat / P if P > 0 else 1.0
        
        return K


def create_rk_calculator(components: list[str]) -> RedlichKwong:
    """Factory function to create RK calculator"""
    return RedlichKwong(components)
