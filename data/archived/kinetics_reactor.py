"""
Kinetics Reactor Module

Implements kinetic reactors with customizable rate expressions:
- CSTR (Continuous Stirred Tank Reactor)
- PFR (Plug Flow Reactor)

Supports:
- Power law kinetics: r = k * prod(C_i^n_i)
- Arrhenius temperature dependence: k = A * exp(-Ea/RT)
- Custom rate expressions via lambda functions
- Isothermal, adiabatic, and jacketed operation modes

Units:
- Concentrations: mol/m3
- Rates: mol/m3/h
- Volume: m3
- Temperature: K
- Activation energy: J/mol
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional
from thermodynamics import StreamState, ThermodynamicsError

# Constants
R = 8.314  # J/mol-K


class KineticsError(Exception):
    """Error in kinetics calculations"""
    pass


@dataclass
class KineticReaction:
    """
    Definition of a kinetic reaction.
    
    Attributes:
        equation: Reaction equation string (e.g., "A + B -> C")
        stoichiometry: Dict of component -> stoichiometric coefficient
                      (negative for reactants, positive for products)
        rate_constant: Pre-exponential factor A [units depend on order]
        activation_energy: Ea [J/mol]
        reaction_orders: Dict of component -> reaction order
        rate_expression: Optional custom rate function f(C, T) -> r
        delta_Hr: Heat of reaction [J/mol of key component]
        key_component: Component for conversion/extent calculation
    """
    equation: str
    stoichiometry: dict[str, float]
    rate_constant: float = 1.0  # A in Arrhenius
    activation_energy: float = 0.0  # Ea [J/mol]
    reaction_orders: dict[str, float] = field(default_factory=dict)
    rate_expression: Optional[Callable] = None  # Custom f(C, T) -> r
    delta_Hr: float = 0.0  # J/mol (negative = exothermic)
    key_component: Optional[str] = None
    
    def __post_init__(self):
        # Set key component to first reactant if not specified
        if self.key_component is None:
            for comp, nu in self.stoichiometry.items():
                if nu < 0:
                    self.key_component = comp
                    break
        
        # Default reaction orders to stoichiometric coefficients
        if not self.reaction_orders:
            self.reaction_orders = {
                comp: abs(nu) 
                for comp, nu in self.stoichiometry.items() 
                if nu < 0  # Only reactants
            }
    
    def rate(self, concentrations: dict[str, float], T: float) -> float:
        """
        Calculate reaction rate [mol/m3/h]
        
        Args:
            concentrations: Component concentrations [mol/m3]
            T: Temperature [K]
            
        Returns:
            Reaction rate (positive = forward)
        """
        # Use custom expression if provided
        if self.rate_expression is not None:
            return self.rate_expression(concentrations, T)
        
        # Power law with Arrhenius
        k = self.rate_constant * math.exp(-self.activation_energy / (R * T))
        
        # Product of concentrations raised to reaction orders
        rate = k
        for comp, order in self.reaction_orders.items():
            C = concentrations.get(comp, 0.0)
            if C < 0:
                C = 0.0
            rate *= C ** order
        
        return rate
    
    def heat_generation(self, rate: float, volume: float) -> float:
        """
        Calculate heat generation rate [J/h]
        
        Args:
            rate: Reaction rate [mol/m3/h]
            volume: Reactor volume [m3]
            
        Returns:
            Heat generation rate [J/h] (negative = heat released)
        """
        return -self.delta_Hr * rate * volume
    
    @classmethod
    def from_equation(cls, equation: str, k: float = 1.0, Ea: float = 0.0,
                      orders: Optional[dict] = None,
                      delta_Hr: float = 0.0) -> 'KineticReaction':
        """
        Create reaction from equation string.
        
        Equation format: "A + 2 B -> C + D" or "A + B <-> C"
        
        Args:
            equation: Reaction equation
            k: Rate constant (pre-exponential factor)
            Ea: Activation energy [J/mol]
            orders: Reaction orders (defaults to stoichiometry)
            delta_Hr: Heat of reaction [J/mol]
            
        Returns:
            KineticReaction instance
        """
        import re
        
        stoich = {}
        
        # Split by -> or <->
        if '->' in equation:
            left, right = equation.split('->')
        elif '<=>' in equation:
            left, right = equation.split('<=>')
        elif '=' in equation:
            left, right = equation.split('=')
        else:
            raise KineticsError(f"Invalid equation format: {equation}")
        
        # Parse reactants (negative coefficients)
        for term in left.split('+'):
            term = term.strip()
            if not term:
                continue
            match = re.match(r'^(\d*\.?\d*)\s*(\w+)$', term)
            if match:
                coeff = float(match.group(1)) if match.group(1) else 1.0
                species = match.group(2)
                stoich[species] = -coeff
        
        # Parse products (positive coefficients)
        for term in right.split('+'):
            term = term.strip()
            if not term:
                continue
            # Remove any parameters after |
            term = term.split('|')[0].strip()
            match = re.match(r'^(\d*\.?\d*)\s*(\w+)$', term)
            if match:
                coeff = float(match.group(1)) if match.group(1) else 1.0
                species = match.group(2)
                stoich[species] = coeff
        
        return cls(
            equation=equation,
            stoichiometry=stoich,
            rate_constant=k,
            activation_energy=Ea,
            reaction_orders=orders or {},
            delta_Hr=delta_Hr
        )


@dataclass
class CSTRResult:
    """Results from CSTR calculation"""
    outlet_concentrations: dict[str, float]  # mol/m3
    outlet_temperature: float  # K
    conversion: float  # Key component conversion
    residence_time: float  # h
    reaction_rates: list[float]  # mol/m3/h per reaction
    heat_generation: float  # J/h
    heat_duty: float  # J/h (positive = cooling needed)


class CSTR:
    """
    Continuous Stirred Tank Reactor (CSTR) model.
    
    Steady-state mole balance:
        F_in * C_in - F_out * C_out + V * sum(nu_i * r_j) = 0
        
    Energy balance (adiabatic):
        F * Cp * (T_out - T_in) = -sum(delta_Hr * r_j * V)
    """
    
    def __init__(self, reactions: list[KineticReaction], volume: float):
        """
        Initialize CSTR.
        
        Args:
            reactions: List of kinetic reactions
            volume: Reactor volume [m3]
        """
        self.reactions = reactions
        self.volume = volume
    
    def solve(self, inlet: StreamState, 
              mode: str = 'isothermal',
              T_set: Optional[float] = None,
              coolant_T: Optional[float] = None,
              UA: float = 0.0,
              Cp_mixture: float = 50000.0) -> CSTRResult:
        """
        Solve CSTR steady state.
        
        Args:
            inlet: Inlet stream state
            mode: 'isothermal', 'adiabatic', or 'jacketed'
            T_set: Set temperature for isothermal [K]
            coolant_T: Coolant temperature for jacketed [K]
            UA: Overall heat transfer coefficient * area [J/h/K]
            Cp_mixture: Mixture heat capacity [J/kmol/K]
            
        Returns:
            CSTRResult with outlet conditions
        """
        # Calculate inlet concentrations [mol/m3]
        # F in kmol/h, need to estimate volumetric flow
        T_in = inlet.T
        P = inlet.P
        
        # Molar volume estimate
        # For ideal gas: V = RT/P with R = 8.314e-5 bar-m3/mol-K
        if inlet.vapor_fraction > 0.5:
            V_molar = 8.314e-5 * T_in / P  # m3/mol for ideal gas
        else:
            V_molar = 0.0001  # m3/mol estimate for liquid (~100 cm3/mol)
        
        Q_in = inlet.F * 1000 * V_molar  # m3/h (F in kmol/h -> mol/h)
        
        C_in = {
            comp: inlet.F * 1000 * z / Q_in 
            for comp, z in inlet.composition.items()
        }
        
        # Initial guess for outlet
        T_out = T_set if T_set else T_in
        C_out = dict(C_in)
        
        tau = self.volume / Q_in  # residence time [h]
        
        # Iterative solution with damping
        damping = 0.3  # Damping factor for stability
        
        for iteration in range(200):
            T_old = T_out
            C_old = dict(C_out)
            
            # Calculate reaction rates at outlet conditions
            rates = [rxn.rate(C_out, T_out) for rxn in self.reactions]
            
            # Update concentrations using implicit formula with damping
            # C_out = C_in + tau * sum(nu_i * r_j(C_out))
            # Use: C_new = C_in + tau * r(C_old), then damp
            
            C_new = {}
            for comp in C_in:
                delta_C = 0.0
                for rxn, r in zip(self.reactions, rates):
                    nu = rxn.stoichiometry.get(comp, 0)
                    delta_C += nu * r * tau
                
                C_new[comp] = max(0, C_in[comp] + delta_C)
            
            # Apply damping
            for comp in C_in:
                C_out[comp] = damping * C_new[comp] + (1 - damping) * C_old[comp]
                C_out[comp] = max(0, C_out[comp])
            
            # Update temperature based on mode
            if mode == 'isothermal':
                T_out = T_set if T_set else T_in
            elif mode == 'adiabatic':
                # Energy balance: F*Cp*(T_out - T_in) = -sum(delta_Hr * r * V)
                Q_rxn = sum(
                    rxn.heat_generation(r, self.volume) 
                    for rxn, r in zip(self.reactions, rates)
                )
                # Cp in J/kmol/K, F in kmol/h
                if inlet.F > 0 and Cp_mixture > 0:
                    T_new = T_in + Q_rxn / (inlet.F * Cp_mixture)
                    # Apply damping to temperature update
                    T_out = damping * T_new + (1 - damping) * T_old
                    # Limit temperature to reasonable range
                    T_out = max(200, min(2000, T_out))
            elif mode == 'jacketed':
                # Energy balance with heat transfer
                Q_rxn = sum(
                    rxn.heat_generation(r, self.volume)
                    for rxn, r in zip(self.reactions, rates)
                )
                if UA > 0 and coolant_T is not None:
                    # Q_transfer = UA * (T - Tc)
                    # F*Cp*(T_out - T_in) = -Q_rxn - UA*(T_out - Tc)
                    if inlet.F > 0 and Cp_mixture > 0:
                        F_Cp = inlet.F * Cp_mixture
                        T_new = (F_Cp * T_in + Q_rxn + UA * coolant_T) / (F_Cp + UA)
                        T_out = damping * T_new + (1 - damping) * T_old
                        T_out = max(200, min(2000, T_out))
            
            # Check convergence
            T_err = abs(T_out - T_old) / max(T_out, 1.0)
            C_err = max(
                abs(C_out.get(c, 0) - C_old.get(c, 0)) / max(C_old.get(c, 0), 1e-10)
                for c in C_in
            )
            
            if T_err < 1e-6 and C_err < 1e-6:
                break
        
        # Calculate conversion of key component
        key_comp = self.reactions[0].key_component if self.reactions else None
        if key_comp and key_comp in C_in and C_in[key_comp] > 0:
            conversion = (C_in[key_comp] - C_out.get(key_comp, 0)) / C_in[key_comp]
        else:
            conversion = 0.0
        
        # Calculate heat duty
        Q_rxn = sum(
            rxn.heat_generation(r, self.volume)
            for rxn, r in zip(self.reactions, rates)
        )
        
        if mode == 'isothermal':
            heat_duty = Q_rxn  # Must remove this heat
        elif mode == 'jacketed' and UA > 0 and coolant_T:
            heat_duty = UA * (T_out - coolant_T)
        else:
            heat_duty = 0.0
        
        return CSTRResult(
            outlet_concentrations=C_out,
            outlet_temperature=T_out,
            conversion=conversion,
            residence_time=tau,
            reaction_rates=rates,
            heat_generation=Q_rxn,
            heat_duty=heat_duty
        )


@dataclass
class PFRResult:
    """Results from PFR calculation"""
    outlet_concentrations: dict[str, float]  # mol/m3
    outlet_temperature: float  # K
    conversion: float  # Key component conversion
    residence_time: float  # h
    profiles: dict[str, list[float]]  # Concentration/temperature profiles along reactor
    volume_profile: list[float]  # Volume positions [m3]
    heat_duty: float  # J/h (total heat transfer, positive = cooling)


class PFR:
    """
    Plug Flow Reactor (PFR) model.
    
    Mole balance along reactor:
        dF_i/dV = sum(nu_ij * r_j)
        
    Energy balance:
        dT/dV = [sum(-delta_Hr_j * r_j) - U*a*(T - Tc)] / (F_total * Cp)
        
    where:
        F_i = molar flow of component i [mol/h]
        V = reactor volume [m3]
        r_j = rate of reaction j [mol/m3/h]
        nu_ij = stoichiometric coefficient
        U = heat transfer coefficient [J/m2/h/K]
        a = specific area [m2/m3]
        Tc = coolant temperature [K]
    """
    
    def __init__(self, reactions: list[KineticReaction], 
                 volume: float,
                 diameter: float = 0.1):
        """
        Initialize PFR.
        
        Args:
            reactions: List of kinetic reactions
            volume: Reactor volume [m3]
            diameter: Reactor diameter [m] (for jacketed)
        """
        self.reactions = reactions
        self.volume = volume
        self.diameter = diameter
        self.length = volume / (math.pi * (diameter/2)**2) if diameter > 0 else 1.0
        # Specific area for cylindrical tube
        self.specific_area = 4 / diameter if diameter > 0 else 0.0
    
    def solve(self, inlet: StreamState,
              mode: str = 'isothermal',
              T_set: Optional[float] = None,
              coolant_T: Optional[float] = None,
              U: float = 0.0,
              Cp_mixture: float = 50000.0,
              n_steps: int = 100) -> PFRResult:
        """
        Solve PFR by numerical integration.
        
        Args:
            inlet: Inlet stream state
            mode: 'isothermal', 'adiabatic', or 'jacketed'
            T_set: Set temperature for isothermal [K]
            coolant_T: Coolant temperature for jacketed [K]
            U: Overall heat transfer coefficient [J/m2/h/K]
            Cp_mixture: Mixture heat capacity [J/kmol/K]
            n_steps: Number of integration steps
            
        Returns:
            PFRResult with outlet conditions and profiles
        """
        T_in = inlet.T
        P = inlet.P
        
        # Calculate inlet molar flows [mol/h]
        F_in = {comp: inlet.F * 1000 * z for comp, z in inlet.composition.items()}
        
        # Volumetric flow estimate [m3/h]
        # For ideal gas: V = RT/P with R = 8.314e-5 bar-m3/mol-K
        if inlet.vapor_fraction > 0.5:
            V_molar = 8.314e-5 * T_in / P  # m3/mol for ideal gas
        else:
            V_molar = 0.0001  # m3/mol for liquid (~100 cm3/mol)
        
        Q_in = inlet.F * 1000 * V_molar  # m3/h
        
        # Integration step size
        dV = self.volume / n_steps
        
        # Initialize state
        F = dict(F_in)  # Current molar flows
        T = T_set if (mode == 'isothermal' and T_set) else T_in
        
        # Storage for profiles
        profiles = {comp: [F_in.get(comp, 0) / 1000] for comp in F}  # Convert back to kmol/h for output
        profiles['T'] = [T]
        volumes = [0.0]
        
        total_heat_transfer = 0.0
        
        # Integrate along reactor
        for step in range(n_steps):
            V_pos = (step + 0.5) * dV  # Midpoint
            
            # Calculate concentrations
            F_total = sum(F.values())
            if F_total > 0:
                # Update volumetric flow for temperature
                V_molar_current = 8.314e-5 * T / P if inlet.vapor_fraction > 0.5 else 0.0001
                Q = F_total * V_molar_current
                C = {comp: F_i / Q for comp, F_i in F.items()}
            else:
                C = {comp: 0.0 for comp in F}
            
            # Calculate reaction rates
            rates = [rxn.rate(C, T) for rxn in self.reactions]
            
            # Update molar flows: dF_i/dV = sum(nu_ij * r_j)
            for comp in F:
                dF = 0.0
                for rxn, r in zip(self.reactions, rates):
                    nu = rxn.stoichiometry.get(comp, 0)
                    dF += nu * r
                F[comp] = max(0, F[comp] + dF * dV)
            
            # Update temperature based on mode
            if mode == 'isothermal':
                T = T_set if T_set else T_in
                # Heat that must be removed
                Q_rxn = sum(
                    -rxn.delta_Hr * r 
                    for rxn, r in zip(self.reactions, rates)
                )
                total_heat_transfer += Q_rxn * dV
                
            elif mode == 'adiabatic':
                # dT/dV = sum(-delta_Hr * r) / (F_total * Cp_molar)
                # where Cp_molar = Cp_mixture / 1000 (J/mol/K)
                Q_rxn = sum(
                    -rxn.delta_Hr * r 
                    for rxn, r in zip(self.reactions, rates)
                )
                if F_total > 0 and Cp_mixture > 0:
                    # Cp_mixture is in J/kmol/K, convert to J/mol/K
                    Cp_molar = Cp_mixture / 1000
                    # F_total is in mol/h
                    dT = Q_rxn / (F_total * Cp_molar)
                    T += dT * dV
                    # Limit temperature change to prevent runaway
                    T = max(200, min(2000, T))
                    
            elif mode == 'jacketed':
                # dT/dV = [sum(-delta_Hr * r) - U*a*(T - Tc)] / (F_total * Cp_molar)
                Q_rxn = sum(
                    -rxn.delta_Hr * r 
                    for rxn, r in zip(self.reactions, rates)
                )
                Q_transfer = U * self.specific_area * (T - coolant_T) if coolant_T else 0
                
                if F_total > 0 and Cp_mixture > 0:
                    Cp_molar = Cp_mixture / 1000  # J/mol/K
                    dT = (Q_rxn - Q_transfer) / (F_total * Cp_molar)
                    T += dT * dV
                    T = max(200, min(2000, T))
                
                total_heat_transfer += Q_transfer * dV
            
            # Store profiles
            for comp in F:
                profiles[comp].append(F[comp] / 1000)  # kmol/h
            profiles['T'].append(T)
            volumes.append((step + 1) * dV)
        
        # Calculate outlet concentrations
        F_total_out = sum(F.values())
        if F_total_out > 0:
            V_molar_out = 0.0831 * T / P if inlet.vapor_fraction > 0.5 else 0.0001
            Q_out = F_total_out * V_molar_out
            C_out = {comp: F_i / Q_out for comp, F_i in F.items()}
        else:
            C_out = {comp: 0.0 for comp in F}
        
        # Calculate conversion
        key_comp = self.reactions[0].key_component if self.reactions else None
        if key_comp and F_in.get(key_comp, 0) > 0:
            conversion = (F_in[key_comp] - F.get(key_comp, 0)) / F_in[key_comp]
        else:
            conversion = 0.0
        
        # Residence time
        tau = self.volume / Q_in if Q_in > 0 else 0.0
        
        return PFRResult(
            outlet_concentrations=C_out,
            outlet_temperature=T,
            conversion=conversion,
            residence_time=tau,
            profiles=profiles,
            volume_profile=volumes,
            heat_duty=total_heat_transfer
        )


def parse_kinetics_params(params: dict) -> tuple[list[KineticReaction], str, dict]:
    """
    Parse kinetics parameters from unit specification.
    
    Expected params format:
        reactions: List of reaction dicts with:
            - equation: "A + B -> C"
            - k or rate_constant: Pre-exponential factor
            - Ea or activation_energy: J/mol
            - orders: {comp: order} (optional)
            - delta_Hr: Heat of reaction J/mol (optional)
        mode: 'isothermal', 'adiabatic', 'jacketed'
        T or T_out: Set temperature [K]
        coolant_T: Coolant temperature [K]
        U: Heat transfer coefficient [J/m2/h/K] or [W/m2/K]
        volume or V: Reactor volume [m3]
        
    Returns:
        (reactions, mode, other_params)
    """
    reactions = []
    
    rxn_list = params.get('reactions', [])
    for rxn in rxn_list:
        equation = rxn.get('equation', '')
        k = rxn.get('k', rxn.get('rate_constant', 1.0))
        Ea = rxn.get('Ea', rxn.get('activation_energy', 0.0))
        orders = rxn.get('orders', rxn.get('reaction_orders', {}))
        delta_Hr = rxn.get('delta_Hr', rxn.get('heat_of_reaction', 0.0))
        
        # Convert Ea from kJ/mol to J/mol if small
        if isinstance(Ea, (int, float)) and 0 < Ea < 500:
            Ea = Ea * 1000  # Assume kJ/mol
        
        # Convert delta_Hr from kJ/mol to J/mol if small magnitude
        if isinstance(delta_Hr, (int, float)) and abs(delta_Hr) < 500:
            delta_Hr = delta_Hr * 1000  # Assume kJ/mol
        
        # Parse string orders if needed
        if isinstance(orders, str):
            orders = {}  # Will default to stoichiometry
        
        try:
            k = float(k)
            Ea = float(Ea)
            delta_Hr = float(delta_Hr)
        except (TypeError, ValueError):
            k, Ea, delta_Hr = 1.0, 0.0, 0.0
        
        kinetic_rxn = KineticReaction.from_equation(
            equation, k=k, Ea=Ea, orders=orders if orders else None, delta_Hr=delta_Hr
        )
        reactions.append(kinetic_rxn)
    
    mode = params.get('mode', 'isothermal')
    
    other = {
        'T_set': params.get('T', params.get('T_out')),
        'coolant_T': params.get('coolant_T', params.get('Tc')),
        'U': params.get('U', params.get('heat_transfer_coeff', 0)),
        'volume': params.get('volume', params.get('V', 1.0)),
        'diameter': params.get('diameter', params.get('D', 0.1)),
        'Cp_mixture': params.get('Cp', params.get('Cp_mixture', 50000)),
    }
    
    # Convert U from W/m2/K to J/m2/h/K if reasonable
    if isinstance(other['U'], (int, float)) and 0 < other['U'] < 10000:
        other['U'] = other['U'] * 3600  # W to J/h
    
    return reactions, mode, other


def create_power_law_reaction(equation: str, 
                               A: float, Ea: float,
                               orders: Optional[dict] = None,
                               delta_Hr: float = 0.0) -> KineticReaction:
    """
    Convenience function to create a power law reaction.
    
    Args:
        equation: Reaction equation
        A: Pre-exponential factor [units depend on overall order]
        Ea: Activation energy [J/mol]
        orders: Reaction orders (defaults to stoichiometry)
        delta_Hr: Heat of reaction [J/mol]
        
    Returns:
        KineticReaction instance
    """
    return KineticReaction.from_equation(equation, k=A, Ea=Ea, 
                                          orders=orders, delta_Hr=delta_Hr)


def create_custom_rate_reaction(equation: str,
                                 rate_func: Callable,
                                 delta_Hr: float = 0.0) -> KineticReaction:
    """
    Create a reaction with a custom rate expression.
    
    Args:
        equation: Reaction equation (for stoichiometry)
        rate_func: Function f(concentrations, T) -> rate [mol/m3/h]
        delta_Hr: Heat of reaction [J/mol]
        
    Returns:
        KineticReaction instance
    """
    rxn = KineticReaction.from_equation(equation, delta_Hr=delta_Hr)
    rxn.rate_expression = rate_func
    return rxn
