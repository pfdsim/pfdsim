"""
Kinetics Reactor Module

Implements kinetics-based reactor models:
- CSTR (Continuous Stirred Tank Reactor)
- PFR (Plug Flow Reactor) with numerical integration

Supports:
- Power law kinetics presets
- Custom kinetics expressions
- Multiple reactions
- Isothermal, adiabatic, and jacketed operation modes
- Heat effects and energy balances

Kinetics Expression Format:
    Power law: r = k * C_A^a * C_B^b
    Arrhenius: k = A * exp(-Ea/(R*T))
    
Custom expressions use Python syntax with:
    - C['species'] for concentrations [kmol/m3]
    - T for temperature [K]
    - P for pressure [bar]
    - k for rate constant (evaluated with Arrhenius)
    - Built-in math functions: exp, log, sqrt, sin, cos, etc.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Callable, Union
from thermodynamics import StreamState, IdealThermodynamics, ThermodynamicsError


# Gas constant
R = 8.314  # J/mol-K


class KineticsError(Exception):
    """Error in kinetics calculations"""
    pass


@dataclass
class Reaction:
    """Definition of a chemical reaction with kinetics"""
    equation: str  # e.g., "A + 2 B -> C + D"
    stoichiometry: dict[str, float] = field(default_factory=dict)  # species: coefficient (- for reactants)
    
    # Kinetics parameters
    kinetics_type: str = "power_law"  # power_law, custom, equilibrium
    
    # Power law: r = k * prod(C_i^order_i)
    rate_constant_A: float = 1.0  # Pre-exponential factor [various units]
    activation_energy: float = 0.0  # Ea [J/mol]
    reaction_orders: dict[str, float] = field(default_factory=dict)  # species: order
    
    # Custom expression: evaluated as Python expression
    custom_expression: str = ""  # e.g., "k * C['A'] * C['B'] / (1 + K_eq * C['C'])"
    custom_params: dict[str, float] = field(default_factory=dict)  # Additional parameters for custom expr
    
    # Heat of reaction
    heat_of_reaction: Optional[float] = None  # kJ/kmol (negative = exothermic)
    reference_T: float = 298.15  # K (for dHr)
    
    def __post_init__(self):
        """Parse reaction equation to stoichiometry if not provided"""
        if not self.stoichiometry and self.equation:
            self.stoichiometry = self._parse_equation(self.equation)
        
        # Default reaction orders to stoichiometry magnitudes for reactants
        if not self.reaction_orders and self.kinetics_type == "power_law":
            for species, coeff in self.stoichiometry.items():
                if coeff < 0:  # Reactant
                    self.reaction_orders[species] = abs(coeff)
    
    def _parse_equation(self, equation: str) -> dict[str, float]:
        """Parse reaction equation into stoichiometry"""
        import re
        
        # Split by -> or <=> or =
        if '->' in equation:
            left, right = equation.split('->', 1)
        elif '<=>' in equation:
            left, right = equation.split('<=>', 1)
        elif '=' in equation:
            left, right = equation.split('=', 1)
        else:
            return {}
        
        stoich = {}
        
        # Parse reactants (negative)
        for term in left.split('+'):
            term = term.strip()
            if not term:
                continue
            match = re.match(r'^(\d*\.?\d*)\s*(\w+)$', term)
            if match:
                coeff = float(match.group(1)) if match.group(1) else 1.0
                species = match.group(2)
                stoich[species] = -coeff
        
        # Parse products (positive)
        for term in right.split('+'):
            term = term.strip()
            if not term:
                continue
            match = re.match(r'^(\d*\.?\d*)\s*(\w+)$', term)
            if match:
                coeff = float(match.group(1)) if match.group(1) else 1.0
                species = match.group(2)
                stoich[species] = coeff
        
        return stoich
    
    def rate_constant(self, T: float) -> float:
        """Calculate rate constant k at temperature T using Arrhenius equation"""
        if self.activation_energy <= 0:
            return self.rate_constant_A
        return self.rate_constant_A * math.exp(-self.activation_energy / (R * T))
    
    def calculate_rate(self, concentrations: dict[str, float], T: float, P: float = 1.0) -> float:
        """
        Calculate reaction rate r [kmol/m3/h]
        
        Args:
            concentrations: Species concentrations [kmol/m3]
            T: Temperature [K]
            P: Pressure [bar]
            
        Returns:
            Reaction rate (positive = forward direction)
        """
        k = self.rate_constant(T)
        
        if self.kinetics_type == "power_law":
            # r = k * prod(C_i^order_i)
            rate = k
            for species, order in self.reaction_orders.items():
                C = concentrations.get(species, 0.0)
                if C < 0:
                    C = 0.0
                # Handle negative concentrations and zero with small power
                if C <= 0 and order != 0:
                    rate = 0.0
                    break
                rate *= C ** order
            return rate
        
        elif self.kinetics_type == "custom":
            return self._evaluate_custom_rate(concentrations, T, P, k)
        
        else:
            raise KineticsError(f"Unknown kinetics type: {self.kinetics_type}")
    
    def _evaluate_custom_rate(self, C: dict[str, float], T: float, P: float, k: float) -> float:
        """Evaluate custom rate expression"""
        if not self.custom_expression:
            return 0.0
        
        # Build evaluation context
        safe_math = {
            'exp': math.exp, 'log': math.log, 'log10': math.log10,
            'sqrt': math.sqrt, 'pow': pow, 'abs': abs,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pi': math.pi, 'e': math.e,
        }
        
        context = {
            'C': C, 'T': T, 'P': P, 'k': k, 'R': R,
            **safe_math,
            **self.custom_params,
        }
        
        try:
            result = eval(self.custom_expression, {"__builtins__": {}}, context)
            return float(result)
        except Exception as e:
            raise KineticsError(f"Error evaluating rate expression '{self.custom_expression}': {e}")


@dataclass 
class ReactorResult:
    """Result from kinetics reactor calculation"""
    outlet_state: StreamState
    heat_duty: float = 0.0  # kJ/h (positive = heat added)
    volume: float = 0.0  # m3
    residence_time: float = 0.0  # h
    conversion: dict[str, float] = field(default_factory=dict)  # species: fractional conversion
    selectivity: dict[str, float] = field(default_factory=dict)  # product: selectivity
    reaction_extents: list[float] = field(default_factory=list)  # kmol/h per reaction
    temperature_profile: list[tuple] = field(default_factory=list)  # (position, T) for PFR
    concentration_profile: list[tuple] = field(default_factory=list)  # (position, {species: C}) for PFR
    warnings: list[str] = field(default_factory=list)


class KineticsReactorBase:
    """Base class for kinetics reactors"""
    
    def __init__(self, thermo: IdealThermodynamics, reactions: list[Reaction], params: dict):
        self.thermo = thermo
        self.reactions = reactions
        self.params = params
    
    def get_param(self, name: str, default=None):
        """Get parameter with default"""
        if name in self.params:
            return self.params[name]
        name_lower = name.lower()
        for k, v in self.params.items():
            if k.lower() == name_lower:
                return v
        return default
    
    def calculate_heat_of_reaction(self, reaction: Reaction, T: float) -> float:
        """Calculate heat of reaction at temperature T [kJ/kmol]"""
        if reaction.heat_of_reaction is not None:
            # Adjust for temperature if Cp data available
            dHr = reaction.heat_of_reaction
            if abs(T - reaction.reference_T) > 10:
                # dH(T) = dH(T_ref) + integral(dCp * dT)
                dCp = 0.0
                for species, nu in reaction.stoichiometry.items():
                    if species in self.thermo.props:
                        props = self.thermo.props[species]
                        # Try to get Cp_avg, with fallback to simple Cp estimate
                        try:
                            if hasattr(props, 'Cp_avg'):
                                dCp += nu * props.Cp_avg(reaction.reference_T, T) / 1000  # J to kJ
                            elif hasattr(props, 'Cp'):
                                dCp += nu * props.Cp(T) / 1000
                        except:
                            pass  # Ignore Cp contribution if unavailable
                dHr += dCp * (T - reaction.reference_T)
            return dHr
        
        # Calculate from formation enthalpies
        dHf = 0.0
        for species, nu in reaction.stoichiometry.items():
            if species in self.thermo.props:
                Hf = getattr(self.thermo.props[species], 'Hf', None)
                if Hf is not None:
                    dHf += nu * Hf  # kJ/mol
        return dHf * 1000  # kJ/kmol
    
    def molar_flow_to_concentration(self, flows: dict[str, float], 
                                     total_flow: float,
                                     T: float, P: float,
                                     phase: str = 'vapor') -> dict[str, float]:
        """Convert molar flows [kmol/h] to concentrations [kmol/m3]"""
        if total_flow <= 0:
            return {s: 0.0 for s in flows}
        
        # Calculate molar volume
        composition = {s: f/total_flow for s, f in flows.items() if total_flow > 0}
        
        if phase == 'vapor' or phase == 'gas':
            # Ideal gas: V = nRT/P
            V_molar = R * T / (P * 1e5) * 1000  # m3/kmol (P in Pa)
        else:
            # Liquid: estimate from MW and density
            MW = self.thermo.mixture_MW(composition)
            rho = 800  # kg/m3 estimate
            V_molar = MW / rho  # m3/kmol
        
        # Volumetric flow rate
        Q = total_flow * V_molar  # m3/h
        
        # Concentrations
        C = {s: f / Q if Q > 0 else 0.0 for s, f in flows.items()}
        return C
    
    def concentration_to_molar_flow(self, C: dict[str, float],
                                     Q: float) -> dict[str, float]:
        """Convert concentrations [kmol/m3] to molar flows [kmol/h]"""
        return {s: c * Q for s, c in C.items()}


class CSTR(KineticsReactorBase):
    """
    Continuous Stirred Tank Reactor
    
    Assumes perfect mixing - outlet composition equals reactor composition.
    
    Design equations:
        For steady state: F_in - F_out + V * r = 0
        Or: tau = (C_in - C_out) / (-r)
    
    Parameters:
        volume: Reactor volume [m3]
        T_out: Operating temperature [K] (isothermal mode)
        mode: 'isothermal', 'adiabatic', or 'jacketed'
        U: Overall heat transfer coefficient [kW/m2-K] (jacketed)
        A_heat: Heat transfer area [m2] (jacketed)
        T_jacket: Jacket temperature [K] (jacketed)
    """
    
    def solve(self, inlet: StreamState) -> ReactorResult:
        """Solve CSTR material and energy balances"""
        V = float(self.get_param('volume', self.get_param('V', 1.0)))
        mode = self.get_param('mode', 'isothermal')
        phase = self.get_param('phase', 'vapor')
        
        # Get inlet conditions
        F_in = inlet.F  # kmol/h
        T_in = inlet.T
        P = inlet.P
        
        # Initialize with inlet composition
        flows_in = inlet.component_flows()
        
        # Operating temperature
        if mode == 'isothermal':
            T_op = self.get_param('T_out', self.get_param('T', T_in))
            if T_op < 200:
                T_op += 273.15  # Celsius to Kelvin
        else:
            T_op = T_in  # Will iterate for adiabatic/jacketed
        
        # Iterative solution for CSTR with damping
        flows_out = dict(flows_in)
        max_iter = 200
        tolerance = 1e-6
        damping = 0.3  # Under-relaxation factor
        
        Q_rxn_total = 0.0
        extents = [0.0] * len(self.reactions)
        
        for iteration in range(max_iter):
            # Calculate outlet concentrations
            F_out = sum(max(0, f) for f in flows_out.values())
            if F_out <= 0:
                break
            
            C = self.molar_flow_to_concentration(flows_out, F_out, T_op, P, phase)
            
            # Calculate volumetric flow rate
            composition = {s: f/F_out for s, f in flows_out.items() if f > 0}
            if phase == 'vapor':
                V_molar = R * T_op / (P * 1e5) * 1000  # m3/kmol
            else:
                MW = self.thermo.mixture_MW(composition)
                V_molar = MW / 800  # Approximate liquid
            Q = F_out * V_molar  # m3/h
            
            # Apply reactions
            flows_calc = dict(flows_in)  # Calculated outlet based on current guess
            Q_rxn = 0.0
            
            for i, rxn in enumerate(self.reactions):
                # Calculate rate at reactor conditions
                rate = rxn.calculate_rate(C, T_op, P)  # kmol/m3/h
                
                # Extent = V * r (kmol/h)
                extent = V * rate
                
                # Limit extent by available reactants (can't consume more than inlet provides)
                for species, nu in rxn.stoichiometry.items():
                    if nu < 0:  # Reactant
                        available = flows_in.get(species, 0)
                        max_extent = available / abs(nu)
                        extent = min(extent, max_extent * 0.99)  # 99% max conversion
                
                extents[i] = extent
                
                # Update calculated flows
                for species, nu in rxn.stoichiometry.items():
                    if species not in flows_calc:
                        flows_calc[species] = 0.0
                    flows_calc[species] += nu * extent
                    flows_calc[species] = max(1e-10, flows_calc[species])
                
                # Heat of reaction
                dHr = self.calculate_heat_of_reaction(rxn, T_op)
                Q_rxn += extent * dHr  # kJ/h
            
            Q_rxn_total = Q_rxn
            
            # Apply damping: new = damping * calculated + (1-damping) * old
            flows_new = {}
            for s in set(flows_calc.keys()) | set(flows_out.keys()):
                f_calc = flows_calc.get(s, 0)
                f_old = flows_out.get(s, 0)
                flows_new[s] = damping * f_calc + (1 - damping) * f_old
            
            # Check convergence
            max_change = max(abs(flows_new[s] - flows_out.get(s, 0)) 
                           for s in flows_new)
            flows_out = flows_new
            
            # Update temperature for non-isothermal modes
            if mode == 'adiabatic':
                # Energy balance: F_in * Cp * (T_out - T_in) + Q_rxn = 0
                # Q_rxn < 0 for exothermic -> T increases
                Cp_mix = self.thermo.mixture_Cp(composition, T_op)  # kJ/kmol-K
                if F_out > 0 and Cp_mix > 0:
                    T_new = T_in - Q_rxn_total / (F_out * Cp_mix)
                    T_op = damping * T_new + (1 - damping) * T_op  # Damp temperature too
                    T_op = max(200, min(2000, T_op))  # Bounds
            
            elif mode == 'jacketed':
                U = float(self.get_param('U', 0.5))  # kW/m2-K
                A = float(self.get_param('A_heat', 10.0))  # m2
                T_jacket = float(self.get_param('T_jacket', 300))
                if T_jacket < 200:
                    T_jacket += 273.15
                
                # Q_heat = U * A * (T_jacket - T_op)
                Cp_mix = self.thermo.mixture_Cp(composition, T_op)  # kJ/kmol-K
                if F_out > 0 and Cp_mix > 0:
                    # Steady state: F*Cp*(T_out - T_in) = -Q_rxn + Q_heat
                    Q_heat = U * A * 3600 * (T_jacket - T_op)  # kJ/h
                    T_new = T_in + (-Q_rxn_total + Q_heat) / (F_out * Cp_mix)
                    T_op = damping * T_new + (1 - damping) * T_op
                    T_op = max(200, min(2000, T_op))
            
            if max_change < tolerance:
                break
        
        # Calculate outlet state
        F_out = sum(max(0, f) for f in flows_out.values())
        composition_out = {s: f/F_out for s, f in flows_out.items() if f > 0}
        
        outlet = self.thermo.calculate_state(T_op, P, F_out, composition_out)
        
        # Calculate conversions
        conversions = {}
        for species, F_in_s in flows_in.items():
            if F_in_s > 0:
                F_out_s = flows_out.get(species, 0)
                conversions[species] = (F_in_s - F_out_s) / F_in_s
        
        # Calculate heat duty
        if mode == 'isothermal':
            H_in = inlet.F * inlet.H if inlet.H else 0
            H_out = outlet.F * outlet.H if outlet.H else 0
            heat_duty = H_out - H_in
        elif mode == 'jacketed':
            U = float(self.get_param('U', 0.5))
            A = float(self.get_param('A_heat', 10.0))
            T_jacket = float(self.get_param('T_jacket', 300))
            if T_jacket < 200:
                T_jacket += 273.15
            heat_duty = U * A * 3600 * (T_jacket - T_op)  # kJ/h
        else:
            heat_duty = 0.0
        
        # Residence time
        Q = F_out * (R * T_op / (P * 1e5) * 1000) if phase == 'vapor' else F_out * 0.05
        tau = V / Q if Q > 0 else 0
        
        return ReactorResult(
            outlet_state=outlet,
            heat_duty=heat_duty,
            volume=V,
            residence_time=tau,
            conversion=conversions,
            reaction_extents=extents,
        )


class PFR(KineticsReactorBase):
    """
    Plug Flow Reactor
    
    Assumes no axial mixing - composition varies along reactor length.
    
    Design equations:
        dF_i/dV = sum_j(nu_ij * r_j)
        dT/dV = (sum_j(-dHr_j * r_j) + Q_heat) / (F_total * Cp_mix)
    
    Parameters:
        volume: Total reactor volume [m3]
        length: Reactor length [m] (optional, for jacketed)
        diameter: Reactor diameter [m] (optional)
        T_out: Outlet temperature [K] (for reverse calculation only)
        mode: 'isothermal', 'adiabatic', 'jacketed'
        U: Heat transfer coefficient [kW/m2-K] (jacketed)
        T_coolant: Coolant temperature [K] (jacketed)
        n_segments: Number of integration segments (default: 50)
    """
    
    def solve(self, inlet: StreamState) -> ReactorResult:
        """Solve PFR using numerical integration"""
        V_total = float(self.get_param('volume', self.get_param('V', 1.0)))
        mode = self.get_param('mode', 'isothermal')
        phase = self.get_param('phase', 'vapor')
        n_segments = int(self.get_param('n_segments', 50))
        
        # Get inlet conditions
        T_in = inlet.T
        P = inlet.P
        flows_in = inlet.component_flows()
        
        # Temperature for isothermal mode
        T_isothermal = self.get_param('T_out', self.get_param('T', T_in))
        if T_isothermal < 200:
            T_isothermal += 273.15
        
        # Jacketed parameters
        if mode == 'jacketed':
            U = float(self.get_param('U', 0.5))  # kW/m2-K
            T_coolant = float(self.get_param('T_coolant', 300))
            if T_coolant < 200:
                T_coolant += 273.15
            diameter = float(self.get_param('diameter', 0.5))  # m
            length = float(self.get_param('length', V_total / (math.pi * (diameter/2)**2)))
            A_per_V = 4 / diameter  # m2/m3 (surface area per volume)
        
        # Initialize state vectors
        dV = V_total / n_segments
        
        # State: [F1, F2, ..., Fn, T]
        species_list = list(flows_in.keys())
        for rxn in self.reactions:
            for s in rxn.stoichiometry:
                if s not in species_list:
                    species_list.append(s)
        
        # Current state
        flows = {s: flows_in.get(s, 0.0) for s in species_list}
        T = T_in if mode != 'isothermal' else T_isothermal
        
        # Profiles for output
        V_profile = [0.0]
        T_profile = [(0.0, T)]
        C_profile = [(0.0, dict(flows))]
        
        # Accumulated heat duty
        Q_total = 0.0
        
        # Integrate along reactor length using RK4
        for segment in range(n_segments):
            V = segment * dV
            
            # Calculate rates at current state
            F_total = sum(max(0, f) for f in flows.values())
            if F_total <= 0:
                break
            
            C = self.molar_flow_to_concentration(flows, F_total, T, P, phase)
            
            # RK4 integration step
            k1_flows, k1_T = self._derivatives(flows, T, P, F_total, C, mode, 
                                               T_isothermal, T_coolant if mode == 'jacketed' else 0,
                                               U if mode == 'jacketed' else 0,
                                               A_per_V if mode == 'jacketed' else 0, phase)
            
            flows_mid1 = {s: flows[s] + 0.5 * dV * k1_flows.get(s, 0) for s in flows}
            T_mid1 = T + 0.5 * dV * k1_T
            F_mid1 = sum(max(0, f) for f in flows_mid1.values())
            C_mid1 = self.molar_flow_to_concentration(flows_mid1, F_mid1, T_mid1, P, phase)
            
            k2_flows, k2_T = self._derivatives(flows_mid1, T_mid1, P, F_mid1, C_mid1, mode,
                                               T_isothermal, T_coolant if mode == 'jacketed' else 0,
                                               U if mode == 'jacketed' else 0,
                                               A_per_V if mode == 'jacketed' else 0, phase)
            
            flows_mid2 = {s: flows[s] + 0.5 * dV * k2_flows.get(s, 0) for s in flows}
            T_mid2 = T + 0.5 * dV * k2_T
            F_mid2 = sum(max(0, f) for f in flows_mid2.values())
            C_mid2 = self.molar_flow_to_concentration(flows_mid2, F_mid2, T_mid2, P, phase)
            
            k3_flows, k3_T = self._derivatives(flows_mid2, T_mid2, P, F_mid2, C_mid2, mode,
                                               T_isothermal, T_coolant if mode == 'jacketed' else 0,
                                               U if mode == 'jacketed' else 0,
                                               A_per_V if mode == 'jacketed' else 0, phase)
            
            flows_end = {s: flows[s] + dV * k3_flows.get(s, 0) for s in flows}
            T_end = T + dV * k3_T
            F_end = sum(max(0, f) for f in flows_end.values())
            C_end = self.molar_flow_to_concentration(flows_end, F_end, T_end, P, phase)
            
            k4_flows, k4_T = self._derivatives(flows_end, T_end, P, F_end, C_end, mode,
                                               T_isothermal, T_coolant if mode == 'jacketed' else 0,
                                               U if mode == 'jacketed' else 0,
                                               A_per_V if mode == 'jacketed' else 0, phase)
            
            # Update state
            for s in flows:
                dF = (k1_flows.get(s, 0) + 2*k2_flows.get(s, 0) + 
                      2*k3_flows.get(s, 0) + k4_flows.get(s, 0)) / 6
                flows[s] = max(0, flows[s] + dV * dF)
            
            dT = (k1_T + 2*k2_T + 2*k3_T + k4_T) / 6
            T = T + dV * dT
            T = max(200, min(2000, T))  # Bounds
            
            # Track heat transfer for jacketed mode
            if mode == 'jacketed':
                Q_segment = U * A_per_V * dV * 3600 * (T_coolant - T)  # kJ/h
                Q_total += Q_segment
            
            # Store profiles
            V_profile.append(V + dV)
            T_profile.append((V + dV, T))
            C_profile.append((V + dV, dict(flows)))
        
        # Calculate outlet state
        F_out = sum(max(0, f) for f in flows.values())
        composition_out = {s: f/F_out for s, f in flows.items() if f > 0 and F_out > 0}
        
        T_out = T if mode != 'isothermal' else T_isothermal
        outlet = self.thermo.calculate_state(T_out, P, F_out, composition_out)
        
        # Calculate conversions
        conversions = {}
        for species, F_in_s in flows_in.items():
            if F_in_s > 0:
                F_out_s = flows.get(species, 0)
                conversions[species] = (F_in_s - F_out_s) / F_in_s
        
        # Calculate heat duty for isothermal mode
        if mode == 'isothermal':
            H_in = inlet.F * inlet.H if inlet.H else 0
            H_out = outlet.F * outlet.H if outlet.H else 0
            heat_duty = H_out - H_in
        elif mode == 'jacketed':
            heat_duty = Q_total
        else:
            heat_duty = 0.0
        
        # Residence time
        Q_vol = F_out * (R * T_out / (P * 1e5) * 1000) if phase == 'vapor' else F_out * 0.05
        tau = V_total / Q_vol if Q_vol > 0 else 0
        
        return ReactorResult(
            outlet_state=outlet,
            heat_duty=heat_duty,
            volume=V_total,
            residence_time=tau,
            conversion=conversions,
            temperature_profile=T_profile,
            concentration_profile=C_profile,
        )
    
    def _derivatives(self, flows: dict, T: float, P: float, 
                     F_total: float, C: dict, mode: str,
                     T_isothermal: float, T_coolant: float,
                     U: float, A_per_V: float, phase: str) -> tuple[dict, float]:
        """
        Calculate derivatives dF/dV and dT/dV
        
        Returns:
            (dF_dict, dT)
        """
        dF = {s: 0.0 for s in flows}
        Q_rxn = 0.0  # Heat generation rate per unit volume
        
        # Sum contributions from each reaction
        for rxn in self.reactions:
            rate = rxn.calculate_rate(C, T, P)  # kmol/m3/h
            
            for species, nu in rxn.stoichiometry.items():
                if species not in dF:
                    dF[species] = 0.0
                dF[species] += nu * rate  # dF/dV = nu * r
            
            # Heat generation
            dHr = self.calculate_heat_of_reaction(rxn, T)  # kJ/kmol
            Q_rxn += -rate * dHr  # kJ/m3/h (negative dHr for exothermic gives positive Q)
        
        # Temperature derivative
        if mode == 'isothermal':
            dT = 0.0
        else:
            # Energy balance: dT/dV = (Q_rxn + Q_heat) / (F_total * Cp)
            composition = {s: f/F_total for s, f in flows.items() if f > 0 and F_total > 0}
            Cp_mix = self.thermo.mixture_Cp(composition, T)  # kJ/kmol-K
            
            if mode == 'jacketed':
                Q_heat = U * A_per_V * 3600 * (T_coolant - T)  # kJ/m3/h
            else:
                Q_heat = 0.0
            
            if F_total > 0 and Cp_mix > 0:
                dT = (Q_rxn + Q_heat) / (F_total * Cp_mix)
            else:
                dT = 0.0
        
        return dF, dT


def parse_reaction_definition(rxn_dict: dict) -> Reaction:
    """
    Parse reaction definition from PFD format to Reaction object.
    
    Supports two formats:
    
    1. Simple (from PFD parser):
        equation: "A + 2 B -> C"
        k: 1e6  # or A
        Ea: 50000
        delta_Hr: -50
        
    2. Structured (for advanced use):
        equation: "A + 2 B -> C"
        kinetics:
            type: power_law | custom
            A: 1e6  # Pre-exponential factor
            Ea: 50000  # Activation energy J/mol
            orders:  # Optional, defaults to stoichiometry
                A: 1
                B: 2
            expression: "k * C['A'] * C['B']"  # For custom type
            params:  # Additional params for custom
                K_eq: 10
        heat_of_reaction: -50  # kJ/mol (optional)
    """
    equation = rxn_dict.get('equation', '')
    
    # Check if using structured format with 'kinetics' sub-dict
    kinetics = rxn_dict.get('kinetics', {})
    
    if kinetics:
        # Structured format
        kinetics_type = kinetics.get('type', 'power_law')
        A = float(kinetics.get('A', kinetics.get('k', 1.0)))
        Ea = float(kinetics.get('Ea', kinetics.get('activation_energy', 0)))
        
        orders = kinetics.get('orders', {})
        if isinstance(orders, dict):
            reaction_orders = {k: float(v) for k, v in orders.items()}
        else:
            reaction_orders = {}
        
        custom_expr = kinetics.get('expression', '')
        custom_params = kinetics.get('params', {})
    else:
        # Simple format (from PFD parser)
        kinetics_type = rxn_dict.get('type', 'power_law')
        
        # Look for rate constant in various forms
        A = rxn_dict.get('A', rxn_dict.get('k', rxn_dict.get('rate_constant', 1.0)))
        if isinstance(A, str):
            A = float(A)
        
        # Look for activation energy
        Ea = rxn_dict.get('Ea', rxn_dict.get('activation_energy', 0))
        if isinstance(Ea, str):
            Ea = float(Ea)
        
        # Parse reaction orders if provided in simple format
        reaction_orders = {}
        orders = rxn_dict.get('orders', {})
        if isinstance(orders, dict):
            reaction_orders = {k: float(v) for k, v in orders.items()}
        
        # Also check for order_Species format (from PFD parser)
        for key, value in rxn_dict.items():
            if key.startswith('order_'):
                species = key[6:]  # Remove 'order_' prefix
                try:
                    reaction_orders[species] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Custom expression
        custom_expr = rxn_dict.get('expression', '')
        custom_params = {}
    
    # Heat of reaction - check multiple forms
    dHr = rxn_dict.get('heat_of_reaction', 
           rxn_dict.get('dHr',
           rxn_dict.get('delta_Hr',
           rxn_dict.get('deltaHr', None))))
    if dHr is not None:
        if isinstance(dHr, str):
            dHr = float(dHr)
        dHr = dHr * 1000  # kJ/mol to kJ/kmol
    
    return Reaction(
        equation=equation,
        kinetics_type=kinetics_type,
        rate_constant_A=A,
        activation_energy=Ea,
        reaction_orders=reaction_orders,
        custom_expression=custom_expr,
        custom_params=custom_params,
        heat_of_reaction=dHr,
    )


def create_kinetics_reactor(reactor_type: str, thermo: IdealThermodynamics,
                            reactions: list[dict], params: dict):
    """
    Factory function to create kinetics reactor.
    
    Args:
        reactor_type: 'CSTR' or 'PFR'
        thermo: Thermodynamics calculator
        reactions: List of reaction definitions (dicts)
        params: Reactor parameters
        
    Returns:
        CSTR or PFR instance
    """
    # Parse reactions
    rxn_objects = [parse_reaction_definition(rxn) for rxn in reactions]
    
    reactor_type = reactor_type.upper()
    
    if reactor_type == 'CSTR':
        return CSTR(thermo, rxn_objects, params)
    elif reactor_type == 'PFR':
        return PFR(thermo, rxn_objects, params)
    else:
        raise KineticsError(f"Unknown reactor type: {reactor_type}")
