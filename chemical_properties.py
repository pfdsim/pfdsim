"""
Chemical Properties Module

Provides lookup and calculation of thermodynamic properties for process simulation.

Features:
- Property lookup from built-in database
- Online lookup from PubChem and NIST when local data missing
- Equation of state calculations
- Property estimation methods (Joback, Lydersen)
- Caching of fetched properties
"""

import json
import math
import os
import urllib.request
import urllib.parse
import urllib.error
import warnings as warnings_module
import tempfile
import uuid
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import re
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_BAR_M3_MOL_K, R_J_MOL_K
else:
    from physical_constants import R_BAR_M3_MOL_K, R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pressure_standards import NORMAL_BOILING_PRESSURE_BAR
else:
    from pressure_standards import NORMAL_BOILING_PRESSURE_BAR


# Gas constant
R = R_J_MOL_K  # J/mol-K
R_BAR = R_BAR_M3_MOL_K  # bar-m3/mol-K

# Memoized py2opsin package version; stamps persistent OPSIN negative-cache
# rows so an OPSIN upgrade automatically invalidates cached failures.
_OPSIN_RUNTIME_VERSION = None

_STP_REFERENCE_TEMPERATURE_K = 298.15


def _inferred_phase_at_stp(props: 'ChemicalProperties') -> Optional[str]:
    """Infer a safe 1-atm reference phase from resolved phase points."""
    def hard_phase_point(name: str) -> bool:
        metadata = (props.property_sources or {}).get(name)
        if not isinstance(metadata, dict):
            return True
        text = ' '.join(
            str(metadata.get(field) or '').strip().lower()
            for field in ('source', 'method', 'notes')
        )
        if any(token in text for token in (
            'estimated', 'estimate', 'provisional', 'nannoolal',
            'joback', 'correlation fallback',
        )):
            return False
        try:
            quality = float(metadata.get('quality', 1.0))
        except (TypeError, ValueError):
            return False
        return math.isfinite(quality) and quality >= 0.90

    try:
        Tt = float(props.Tt) if props.Tt is not None else None
        Pt = float(props.Pt) if props.Pt is not None else None
        Tm = float(props.Tm) if props.Tm is not None else None
        Tb = float(props.Tb) if props.Tb is not None else None
    except (TypeError, ValueError):
        return None

    if (
        Tt is not None
        and Pt is not None
        and hard_phase_point('Tt')
        and hard_phase_point('Pt')
        and Pt >= NORMAL_BOILING_PRESSURE_BAR - 1.0e-9
    ):
        if _STP_REFERENCE_TEMPERATURE_K >= Tt:
            return 'gas'
        return None
    if Tm is not None and Tm > _STP_REFERENCE_TEMPERATURE_K:
        return 'solid'
    if Tb is not None and Tb <= _STP_REFERENCE_TEMPERATURE_K:
        return 'gas'
    if (
        Tm is not None
        and Tm <= _STP_REFERENCE_TEMPERATURE_K
        and Tb is not None
        and Tb > _STP_REFERENCE_TEMPERATURE_K
    ):
        return 'liquid'
    return None


@dataclass
class SmilesResolution:
    """Resolved molecular structure with provenance."""
    smiles: str
    source: str
    method: str
    quality: float = 1.0
    notes: str = ""
    identifier: str = ""
    data: dict = field(default_factory=dict)

    def property_source(self) -> dict:
        return {
            'source': self.source,
            'method': self.method,
            'quality': self.quality,
            'notes': self.notes,
        }


@dataclass
class ChemicalProperties:
    """Properties for a single chemical species"""
    symbol: str
    name: str
    formula: str
    CAS: str = ""
    smiles: Optional[str] = None
    MW: float = 0.0  # g/mol
    Tc: Optional[float] = None  # K (critical temperature)
    Pc: Optional[float] = None  # bar (critical pressure)
    Vc: Optional[float] = None  # cm3/mol (critical volume)
    Zc: Optional[float] = None  # critical compressibility factor
    omega: Optional[float] = None  # acentric factor
    mc_c1: Optional[float] = None  # Mathias-Copeman alpha parameter c1
    mc_c2: Optional[float] = None  # Mathias-Copeman alpha parameter c2
    mc_c3: Optional[float] = None  # Mathias-Copeman alpha parameter c3
    kappa1: Optional[float] = None  # PRSV alpha parameter kappa1
    kappa2: Optional[float] = None  # PRSV2 alpha parameter kappa2
    kappa3: Optional[float] = None  # PRSV2 alpha parameter kappa3
    twu_l: Optional[float] = None  # Twu alpha parameter L
    twu_m: Optional[float] = None  # Twu alpha parameter M
    twu_n: Optional[float] = None  # Twu alpha parameter N
    twu_c: Optional[float] = None  # Twu volume-translation parameter c
    henry_Hcp: Optional[float] = None  # Hcp at 298.15 K [mol/(m3*Pa)]
    henry_B: Optional[float] = None  # d(ln Hcp)/d(1/T) [K]
    henry_Tmin: Optional[float] = None  # Full-quality Henry correlation lower bound [K]
    henry_Tmax: Optional[float] = None  # Full-quality Henry correlation upper bound [K]
    henry_Vinf: Optional[float] = None  # Aqueous infinite-dilution partial molar volume [cm3/mol]
    henry_Vinf_uncertainty: Optional[float] = None  # Standard uncertainty [cm3/mol]
    Tb: Optional[float] = None  # K (normal boiling point)
    Tm: Optional[float] = None  # K (melting point)
    Hf: Optional[float] = None  # kJ/mol (standard enthalpy of formation)
    Gf: Optional[float] = None  # kJ/mol (standard Gibbs energy of formation)
    S: Optional[float] = None  # J/mol-K (standard molar entropy)
    Hf_liquid: Optional[float] = None  # kJ/mol (liquid standard enthalpy of formation)
    Gf_liquid: Optional[float] = None  # kJ/mol (liquid standard Gibbs energy of formation)
    S_liquid: Optional[float] = None  # J/mol-K (liquid standard molar entropy)
    Hf_solid: Optional[float] = None  # kJ/mol (solid standard enthalpy of formation)
    Gf_solid: Optional[float] = None  # kJ/mol (solid standard Gibbs energy of formation)
    S_solid: Optional[float] = None  # J/mol-K (solid standard molar entropy)
    Hcomb: Optional[float] = None  # kJ/mol (net heat of combustion)
    Hcomb_gross: Optional[float] = None  # kJ/mol (gross heat of combustion)
    Cp_coeffs: list = field(default_factory=list)  # Ideal-gas Cp polynomial, when explicitly available
    Cp_liquid: Optional[float] = None  # J/mol-K (liquid heat capacity)
    Cp_solid: Optional[float] = None  # J/mol-K (solid heat capacity reference/constant)
    rho_solid: Optional[float] = None  # kg/m3 (explicit constant solid mass density)
    Vm_solid: Optional[float] = None  # m3/kmol (explicit constant solid molar volume)
    dipole_moment: Optional[float] = None  # permanent gas-phase dipole [Debye]
    radius_of_gyration: Optional[float] = None  # mass-weighted Rg [angstrom]
    modified_radius_of_gyration: Optional[float] = None  # Thompson R' [angstrom]
    hoc_eta: Optional[float] = None  # HOC pure/self association parameter
    solid_material_form: str = "unspecified"
    solid_polymorph: str = ""
    property_correlations: dict = field(default_factory=dict)
    vapor_dimerization: Optional[dict] = None
    uniquac_r: Optional[float] = None  # UNIQUAC volume parameter
    uniquac_q: Optional[float] = None  # UNIQUAC surface-area parameter
    Hvap: Optional[float] = None  # kJ/mol (heat of vaporization at Tb)
    Hfus: Optional[float] = None  # kJ/mol (heat of fusion)
    fusion_transitions: list[dict] = field(default_factory=list)
    melting_transitions: list[dict] = field(default_factory=list)
    Hsub: Optional[float] = None  # kJ/mol (heat of sublimation)
    phase_at_STP: str = "unknown"
    critical_properties_unavailable: bool = False
    source: str = "local"  # local, pubchem, nist, estimated
    lookup_warnings: list[str] = field(default_factory=list)
    property_sources: dict = field(default_factory=dict)
    # Antoine equation: log10(P_bar) = A - B/(C + T_C)
    # T in Celsius, P in bar
    antoine_A: Optional[float] = None
    antoine_B: Optional[float] = None
    antoine_C: Optional[float] = None
    antoine_Tmin: Optional[float] = None  # K
    antoine_Tmax: Optional[float] = None  # K
    antoine_source: Optional[str] = None
    Tt: Optional[float] = None  # K (triple-point temperature)
    Pt: Optional[float] = None  # bar (triple-point pressure)
    def _resolver_props(self) -> dict:
        return asdict(self)

    def ideal_gas_cp_kernel(self):
        """Return and retain this component's executable ideal-gas Cp curve."""
        kernel = getattr(self, '_ideal_gas_cp_kernel', None)
        if kernel is None:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            kernel = get_property_resolver().resolve_ideal_gas_cp_kernel(
                self.symbol,
                self._resolver_props(),
            )
            self._ideal_gas_cp_kernel = kernel
        return kernel
    
    def Cp(self, T: float) -> float:
        """
        Ideal-gas heat capacity at temperature T [K].

        Delegates to PropertyResolver so this path uses the same source order
        as every other ideal-gas Cp request.
        """
        try:
            kernel = self.ideal_gas_cp_kernel()
            return kernel.cp(T) if kernel is not None else 3.5 * R
        except Exception:
            return 3.5 * R
    
    def Cp_avg(self, T1: float, T2: float) -> float:
        """Average Cp between T1 and T2 [J/mol-K]"""
        if abs(T2 - T1) < 0.1:
            return self.Cp((T1 + T2) / 2)
        return self.delta_H(T1, T2) / (T2 - T1)
    
    def delta_H(self, T1: float, T2: float) -> float:
        """
        Enthalpy change from T1 to T2 [J/mol]
        Integral of Cp from T1 to T2
        """
        if abs(T2 - T1) < 1e-12:
            return 0.0
        try:
            kernel = self.ideal_gas_cp_kernel()
            if kernel is not None:
                return kernel.delta_h(T1, T2)
        except Exception:
            pass

        steps = 16
        h = (T2 - T1) / steps
        total = self.Cp(T1) + self.Cp(T2)
        for index in range(1, steps):
            weight = 4 if index % 2 else 2
            total += weight * self.Cp(T1 + index * h)
        return total * h / 3.0

    def delta_S(self, T1: float, T2: float) -> float:
        """Ideal-gas entropy change integral Cp/T dT [J/mol-K]."""
        if abs(T2 - T1) < 1e-12:
            return 0.0
        try:
            kernel = self.ideal_gas_cp_kernel()
            if kernel is not None:
                return kernel.delta_s(T1, T2)
        except Exception:
            pass
        steps = 32
        h = (T2 - T1) / steps
        total = self.Cp(T1) / T1 + self.Cp(T2) / T2
        for index in range(1, steps):
            temperature = T1 + index * h
            weight = 4 if index % 2 else 2
            total += weight * self.Cp(temperature) / temperature
        return total * h / 3.0

    def solid_cp_kernel(self):
        """Return and retain this component's executable solid Cp curve."""
        kernel = getattr(self, '_solid_cp_kernel', None)
        if kernel is None:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            kernel = get_property_resolver().resolve_solid_cp_kernel(
                self.symbol,
                self._resolver_props(),
            )
            self._solid_cp_kernel = kernel
        return kernel

    def solid_heat_capacity(self, T: float) -> float:
        """Solid constant-pressure heat capacity [J/mol-K]."""
        kernel = self.solid_cp_kernel()
        if kernel is None:
            raise ValueError(f"No solid heat-capacity kernel is available for {self.symbol!r}")
        return kernel.cp(T)

    def solid_delta_H(self, T1: float, T2: float) -> float:
        """Solid sensible enthalpy increment [J/mol]."""
        kernel = self.solid_cp_kernel()
        if kernel is None:
            raise ValueError(f"No solid heat-capacity kernel is available for {self.symbol!r}")
        return kernel.delta_h(T1, T2)

    def solid_delta_S(self, T1: float, T2: float) -> float:
        """Solid sensible entropy increment [J/mol-K]."""
        kernel = self.solid_cp_kernel()
        if kernel is None:
            raise ValueError(f"No solid heat-capacity kernel is available for {self.symbol!r}")
        return kernel.delta_s(T1, T2)
    
    def antoine_covers_temperature(self, T: float, tolerance: float = 1e-9) -> bool:
        """Return True when the stored Antoine fit is valid at T."""
        if self.antoine_A is None or self.antoine_B is None or self.antoine_C is None:
            return False
        if self.antoine_Tmin is not None and T < self.antoine_Tmin - tolerance:
            return False
        if self.antoine_Tmax is not None and T > self.antoine_Tmax + tolerance:
            return False
        return True

    def Psat_antoine(self, T: float, allow_extrapolation: bool = False) -> Optional[float]:
        """
        Vapor pressure using Antoine equation [bar]
        
        Coefficients are expected in form: log10(P_bar) = A - B/(C + T_C)
        where T_C is temperature in Celsius (T_C = T - 273.15)
        
        Note: Many older sources use mmHg. NIST provides bar-based coefficients.
        """
        if self.antoine_A is None or self.antoine_B is None or self.antoine_C is None:
            return None
        if not allow_extrapolation and not self.antoine_covers_temperature(T):
            return None
        
        T_C = T - 273.15  # Convert K to C
        
        try:
            # Antoine: log10(P_bar) = A - B/(C + T_C)
            log10_P_bar = self.antoine_A - self.antoine_B / (self.antoine_C + T_C)
            P_bar = 10 ** log10_P_bar
            return P_bar
        except (ValueError, ZeroDivisionError):
            return None
    
    def Psat(self, T: float) -> Optional[float]:
        """
        Vapor pressure [bar].

        Delegates to PropertyResolver so every Psat path uses the same
        priority chain.
        """
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver, PropertyResolutionError
            else:
                from property_resolver import get_property_resolver, PropertyResolutionError
            return get_property_resolver().resolve_vapor_pressure(
                self.symbol,
                T,
                self._resolver_props(),
            ).value
        except (ImportError, PropertyResolutionError):
            return None
    
    def Hvap_at_T(self, T: float) -> Optional[float]:
        """
        Heat of vaporization at temperature T [kJ/mol]
        Delegates temperature-dependent resolution, including Watson scaling,
        to PropertyResolver.
        """
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_hvap(
                self.symbol,
                self._resolver_props(),
                T=T,
            )
            if result.value is not None:
                return result.value
        except Exception:
            hvap = self.Hvap

        return hvap
    
    def Z_factor(self, T: float, P: float) -> float:
        """
        Compressibility factor using Peng-Robinson EOS
        Returns Z for vapor phase (largest real root)
        """
        if self.Tc is None or self.Pc is None or self.omega is None:
            return 1.0  # Ideal gas
        
        Tr = T / self.Tc
        Pr = P / self.Pc
        
        # Peng-Robinson parameters
        kappa = 0.37464 + 1.54226*self.omega - 0.26992*self.omega**2
        alpha = (1 + kappa * (1 - math.sqrt(Tr)))**2
        
        a = 0.45724 * alpha * Pr / Tr**2
        b = 0.07780 * Pr / Tr
        
        # Cubic: Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0
        A, B = a, b
        
        # Solve cubic using Cardano's method
        p = A - 3*B**2 - 2*B - (1-B)**2/3
        q = -A*B + B**2 + B**3 + 2*(1-B)**3/27 - (1-B)*(A-3*B**2-2*B)/3
        
        discriminant = (q/2)**2 + (p/3)**3
        
        if discriminant > 0:
            # One real root
            u = (-q/2 + math.sqrt(discriminant))**(1/3)
            v = (-q/2 - math.sqrt(discriminant))**(1/3) if -q/2 - math.sqrt(discriminant) >= 0 else \
                -abs(-q/2 - math.sqrt(discriminant))**(1/3)
            Z = u + v + (1-B)/3
        else:
            # Three real roots - return largest (vapor)
            r = math.sqrt(-(p/3)**3)
            theta = math.acos(-q/(2*r))
            Z = 2 * r**(1/3) * math.cos(theta/3) + (1-B)/3
        
        return max(Z, B + 0.001)  # Z must be > B
    
    def density(self, T: float, P: float) -> float:
        """
        Molar density [mol/m3] using equation of state
        """
        if self.is_vapor(T, P) is False:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .property_resolver import get_property_resolver
                else:
                    from property_resolver import get_property_resolver
                rho_mol_dm3 = get_property_resolver().resolve_liquid_molar_density(
                    self.symbol,
                    T,
                    self._resolver_props(),
                ).value
                return rho_mol_dm3 * 1000.0
            except Exception:
                pass
        Z = self.Z_factor(T, P)
        return P / (Z * R_BAR * T) * 1e-3  # mol/m3

    def solid_mass_density(self, T: float) -> float:
        """Intrinsic solid mass density at ``T`` [kg/m3]."""
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .property_resolver import get_property_resolver
        else:
            from property_resolver import get_property_resolver
        return get_property_resolver().resolve_solid_mass_density(
            self.symbol,
            T,
            self._resolver_props(),
        ).value

    def solid_molar_volume(self, T: float) -> float:
        """Solid molar volume at ``T`` [m3/kmol]."""
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .property_resolver import get_property_resolver
        else:
            from property_resolver import get_property_resolver
        return get_property_resolver().resolve_solid_molar_volume(
            self.symbol,
            T,
            self._resolver_props(),
        ).value

    def solid_molar_density(self, T: float) -> float:
        """Solid molar density at ``T`` [mol/dm3]."""
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .property_resolver import get_property_resolver
        else:
            from property_resolver import get_property_resolver
        return get_property_resolver().resolve_solid_molar_density(
            self.symbol,
            T,
            self._resolver_props(),
        ).value
    
    def is_vapor(self, T: float, P: float) -> Optional[bool]:
        """
        Determine if species is vapor at given T (K) and P (bar)
        Returns None if cannot determine (supercritical)
        """
        if self.Tc is not None and self.Pc is not None:
            if T > self.Tc and P > self.Pc:
                return None  # Supercritical
        
        Psat = self.Psat(T)
        if Psat is None:
            if self.Tb is not None:
                return T > self.Tb * (1 + 0.1 * math.log(max(P, 0.01)))
            return None
        
        return P < Psat
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class OnlinePropertyFetcher:
    """Fetches chemical properties from online databases"""
    
    PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    PUBCHEM_COMPONENT_CACHE_VERSION = 6

    @classmethod
    def _pubchem_component_cache_key(
        cls,
        identifier: str,
        *,
        cas_lookup: bool = False,
    ) -> str:
        family = 'pubchem_component_cas' if cas_lookup else 'pubchem_component'
        return f'{family}_v{cls.PUBCHEM_COMPONENT_CACHE_VERSION}_{identifier}'
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".pfd_cache"
        self._cache = {}
        self._sqlite_cache_state = None
        self._legacy_component_cache_migrated = False
        self.lookup_rules = self._load_lookup_rules()

    def _load_lookup_rules(self) -> dict:
        rules_path = Path(__file__).parent / "data" / "property_lookup_rules.json"
        try:
            with open(rules_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Could not load property lookup rules from {rules_path}") from e

    def _sqlite_cache(self):
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .property_resolution.runtime_cache import (
                        LEGACY_ONLINE_COMPONENT_CACHE_DIR,
                        SQLiteJSONCache,
                        runtime_cache_path_for_legacy_directory,
                    )
        else:
            from property_resolution.runtime_cache import (
                        LEGACY_ONLINE_COMPONENT_CACHE_DIR,
                        SQLiteJSONCache,
                        runtime_cache_path_for_legacy_directory,
                    )

        path = runtime_cache_path_for_legacy_directory(
            self.cache_dir,
            default_legacy_directory=LEGACY_ONLINE_COMPONENT_CACHE_DIR,
        )
        state = self._sqlite_cache_state
        if state is not None and state[0] == path:
            return state[1]
        cache = SQLiteJSONCache(path, 'online_property_fetcher')
        cache.migrate_json_directory(
            self.cache_dir,
            migration_name='online-property-fetcher-cache',
        )
        self._migrate_legacy_pubchem_component_payloads(cache)
        self._sqlite_cache_state = path, cache
        return cache

    def _migrate_legacy_pubchem_component_payloads(self, cache) -> None:
        """Carry identity/hydrated data forward without legacy scalar parsing."""
        if self._legacy_component_cache_migrated:
            return
        legacy_prefixes = (
            ('pubchem_component_v2_', 'pubchem_component_v4_'),
            ('pubchem_component_cas_v2_', 'pubchem_component_cas_v4_'),
            ('pubchem_component_v3_', 'pubchem_component_v4_'),
            ('pubchem_component_cas_v3_', 'pubchem_component_cas_v4_'),
        )
        for old_prefix, new_prefix in legacy_prefixes:
            for key, payload in cache.items(prefix=old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if cache.get(new_key) is not None:
                    continue
                cache.set(
                    new_key,
                    self._without_legacy_pubchem_scalar_fields(payload),
                )
        self._legacy_component_cache_migrated = True

    @staticmethod
    def _without_legacy_pubchem_scalar_fields(payload: dict) -> dict:
        """Strip only scalars owned by the removed first-value PUG parser."""
        sanitized = dict(payload)
        sources = dict(sanitized.get('property_sources') or {})
        removed = set()
        for field_name in ('Tb', 'Tm', 'Tc', 'Pc'):
            source = sources.get(field_name) or {}
            if str(source.get('method') or '').lower() != 'pubchem_pug_view':
                continue
            sanitized[field_name] = None
            sources.pop(field_name, None)
            removed.add(field_name)
        sanitized['property_sources'] = sources
        if removed.intersection({'Tb', 'Tm'}):
            sanitized['phase_at_STP'] = None
        sanitized['lookup_warnings'] = [
            warning
            for warning in (sanitized.get('lookup_warnings') or [])
            if 'pubchem_pug_view' not in str(warning).lower()
        ]
        return sanitized
    
    def _get_cached(self, identifier: str) -> Optional[dict]:
        """Check memory and shared SQLite for previously fetched data."""
        if identifier in self._cache:
            return self._cache[identifier]

        try:
            data = self._sqlite_cache().get(identifier)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return None
        if data is not None:
            self._cache[identifier] = data
        return data
    
    def _save_cache(self, identifier: str, data: dict):
        """Atomically save fetched data to shared SQLite storage."""
        self._cache[identifier] = data
        try:
            self._sqlite_cache().set(identifier, data)
        except (OSError, sqlite3.Error, TypeError, ValueError):
            pass

    def _save_missing_cache(self, identifier: str, reason: str = 'not_found'):
        """Save a negative lookup result so failed online queries are not repeated."""
        self._save_cache(identifier, {'_missing': True, 'reason': reason})

    def save_resolved_pubchem_cache(
        self,
        identifier: str,
        props: ChemicalProperties,
        cas_lookup: bool = False,
    ) -> None:
        """Persist the hydrated online component so stale estimates do not linger."""
        keys = {
            self._pubchem_component_cache_key(identifier),
            self._pubchem_component_cache_key(props.symbol),
        }
        if props.CAS:
            keys.add(self._pubchem_component_cache_key(props.CAS))
        if cas_lookup:
            keys.add(self._pubchem_component_cache_key(
                identifier.strip().replace(' ', ''),
                cas_lookup=True,
            ))
            if props.CAS:
                keys.add(self._pubchem_component_cache_key(
                    props.CAS,
                    cas_lookup=True,
                ))
        payload = props.to_dict()
        for key in keys:
            self._save_cache(key, payload)

    @staticmethod
    def _structure_from_cached_pubchem(data: Optional[dict]) -> Optional[dict]:
        """Return cached structure data from an older full PubChem property cache."""
        if not data or data.get('_missing'):
            return None
        smiles = (
            data.get('smiles')
            or data.get('ConnectivitySMILES')
            or data.get('CanonicalSMILES')
            or data.get('IsomericSMILES')
        )
        if not smiles:
            return None
        return {
            'smiles': smiles,
            'connectivity_smiles': data.get('ConnectivitySMILES') or smiles,
            'canonical_smiles': data.get('CanonicalSMILES') or smiles,
            'isomeric_smiles': data.get('IsomericSMILES'),
            'inchi': data.get('InChI'),
            'name': data.get('name') or data.get('IUPACName'),
            'formula': data.get('formula') or data.get('MolecularFormula'),
            'source': 'pubchem_cache',
        }

    @staticmethod
    def _is_missing_cache(data: Optional[dict]) -> bool:
        return bool(data and data.get('_missing'))

    @staticmethod
    def _is_transient_lookup_error(error: Exception) -> bool:
        """Return True when a lookup failed for a retryable network reason."""
        if isinstance(error, urllib.error.HTTPError):
            return error.code in (408, 429) or error.code >= 500
        return isinstance(error, (urllib.error.URLError, TimeoutError, OSError))

    def fetch_from_textbook(self, identifier: str) -> Optional[ChemicalProperties]:
        """Fetch a chemical from the extracted textbook table without network IO."""
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .textbook_properties import get_textbook_property_library
            else:
                from textbook_properties import get_textbook_property_library
        except ImportError:
            return None

        entry = get_textbook_property_library().get(identifier)
        if not entry:
            return None

        symbol = entry.get('formula') or entry.get('canonical_name') or identifier
        props = ChemicalProperties(
            symbol=symbol,
            name=entry.get('name') or entry.get('canonical_name') or identifier,
            formula=entry.get('formula') or '',
            MW=entry.get('MW') or 0.0,
            Tc=entry.get('Tc'),
            Pc=entry.get('Pc'),
            Vc=entry.get('Vc'),
            omega=entry.get('omega'),
            Zc=entry.get('Zc'),
            Tb=entry.get('Tb'),
            Hvap=entry.get('Hvap'),
            source='textbook',
            antoine_A=entry.get('antoine_A'),
            antoine_B=entry.get('antoine_B'),
            antoine_C=entry.get('antoine_C'),
            antoine_Tmin=entry.get('antoine_Tmin'),
            antoine_Tmax=entry.get('antoine_Tmax'),
        )
        if props.Tb is not None and props.Tb <= float(self.lookup_rules['reference_temperature_K']):
            props.phase_at_STP = 'gas'
        return props

    def fetch_from_perry(self, identifier: str) -> Optional[ChemicalProperties]:
        """Fetch a chemical from the extracted Perry tables without network IO."""
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .perry_properties import get_perry_property_library
            else:
                from perry_properties import get_perry_property_library
        except ImportError:
            return None

        library = get_perry_property_library()
        entry = library.get(identifier, expand_identity=False)
        if not entry:
            return None

        critical = entry.get('critical_constants') or {}
        formation = entry.get('formation_properties') or {}
        mw = (
            critical.get('molecular_weight')
            or formation.get('molecular_weight')
            or 0.0
        )
        formula = entry.get('formula') or (entry.get('formulas') or [''])[0]
        tb = library.normal_boiling_point_K(identifier)
        tm = library.normal_melting_point_K(identifier)
        hvap = library.heat_of_vaporization_kJ_per_mol(identifier)
        hfus = library.heat_of_fusion_kJ_per_mol(identifier)
        props = ChemicalProperties(
            symbol=formula or entry.get('name') or identifier,
            name=entry.get('name') or identifier,
            formula=formula or '',
            CAS=entry.get('cas') or '',
            MW=mw,
            Tc=critical.get('Tc_K'),
            Pc=critical.get('Pc_MPa') * 10.0 if critical.get('Pc_MPa') is not None else None,
            Vc=critical.get('Vc_m3_per_kmol') * 1000.0 if critical.get('Vc_m3_per_kmol') is not None else None,
            Zc=critical.get('Zc'),
            omega=critical.get('omega'),
            Tb=tb.value if tb else None,
            Tm=tm.value if tm else None,
            Hf=(
                formation.get('Hf_ideal_gas_J_per_kmol') / 1.0e6
                if formation.get('Hf_ideal_gas_J_per_kmol') is not None
                else None
            ),
            Gf=(
                formation.get('Gf_ideal_gas_J_per_kmol') / 1.0e6
                if formation.get('Gf_ideal_gas_J_per_kmol') is not None
                else None
            ),
            S=(
                formation.get('S_ideal_gas_J_per_kmol_K') / 1000.0
                if formation.get('S_ideal_gas_J_per_kmol_K') is not None
                else None
            ),
            Hcomb=(
                formation.get('net_Hcomb_J_per_kmol') / 1.0e6
                if formation.get('net_Hcomb_J_per_kmol') is not None
                else None
            ),
            Cp_coeffs=[],
            Hvap=hvap.value if hvap else None,
            Hfus=hfus.value if hfus else None,
            source='perry',
        )
        for key in ('Tc', 'Pc', 'Vc', 'Zc', 'omega', 'Tb', 'Tm', 'Hf', 'Gf', 'S', 'Hcomb', 'Hvap', 'Hfus'):
            if getattr(props, key) is not None:
                props.property_sources[key] = {
                    'source': 'Perry 9th',
                    'method': 'perry_local_fetch',
                    'quality': 0.96,
                    'notes': 'Hydrated from extracted Perry tables',
                }
        if props.Tb is not None and props.Tb <= float(self.lookup_rules['reference_temperature_K']):
            props.phase_at_STP = 'gas'
        return props
    
    def fetch_from_pubchem(self, identifier: str) -> Optional[ChemicalProperties]:
        """
        Fetch properties from PubChem by name, formula, or CAS number
        
        Args:
            identifier: Chemical name, formula, or CAS number
            
        Returns:
            ChemicalProperties or None if not found
        """
        # Check cache first
        cache_key = self._pubchem_component_cache_key(identifier)
        cached = self._get_cached(cache_key)
        if cached:
            if self._is_missing_cache(cached):
                return None
            result = ChemicalProperties(**cached)
            self._repair_cached_pubchem_sources(result)
            if not result.CAS:
                self._enrich_pubchem_cas(result, cache_key)
            return result
        
        try:
            # First, get the CID (PubChem compound ID)
            search_url = f"{self.PUBCHEM_BASE}/compound/name/{urllib.parse.quote(identifier)}/cids/JSON"
            
            req = urllib.request.Request(search_url)
            req.add_header('User-Agent', 'PFD-Editor/1.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                cid = data['IdentifierList']['CID'][0]
            
            # Now fetch properties
            props_url = f"{self.PUBCHEM_BASE}/compound/cid/{cid}/property/" \
                       f"MolecularFormula,MolecularWeight,IUPACName,ConnectivitySMILES,CanonicalSMILES,IsomericSMILES/JSON"
            
            with urllib.request.urlopen(props_url, timeout=10) as response:
                props_data = json.loads(response.read().decode())
                props = props_data['PropertyTable']['Properties'][0]
            
            # Try to get additional properties from PubChem
            name = props.get('IUPACName', identifier)
            formula = props.get('MolecularFormula', identifier)
            mw = props.get('MolecularWeight', 0)
            smiles = (
                props.get('ConnectivitySMILES')
                or props.get('CanonicalSMILES')
                or props.get('IsomericSMILES')
            )
            
            # Create properties object
            result = ChemicalProperties(
                symbol=identifier,
                name=name,
                formula=formula,
                smiles=smiles,
                MW=float(mw) if mw else 0,
                source='pubchem'
            )

            self._apply_pubchem_identity_properties(result, cid)
            direct_fields = self._property_field_status(result)
            lookup_warnings = list(result.lookup_warnings)
            
            # Try to estimate missing properties
            result = self._estimate_missing_properties(result)
            result.lookup_warnings = self._dedupe_lookup_warnings(
                lookup_warnings + result.lookup_warnings
                + self._build_lookup_warnings(identifier, result, direct_fields)
            )
            for warning in result.lookup_warnings:
                warnings_module.warn(warning, RuntimeWarning, stacklevel=2)
            
            # Cache the result
            self._save_cache(cache_key, result.to_dict())
            
            return result
            
        except Exception as e:
            print(f"PubChem lookup failed for '{identifier}': {e}")
            if not self._is_transient_lookup_error(e):
                self._save_missing_cache(cache_key, str(e))
            return None

    def _get_pubchem_cid(self, identifier: str) -> Optional[int]:
        search_url = f"{self.PUBCHEM_BASE}/compound/name/{urllib.parse.quote(identifier)}/cids/JSON"
        req = urllib.request.Request(search_url)
        req.add_header('User-Agent', 'PFD-Editor/1.0')
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        cids = data.get('IdentifierList', {}).get('CID') or []
        return int(cids[0]) if cids else None

    def _enrich_pubchem_cas(self, props: ChemicalProperties, cache_key: str) -> None:
        """Fill a missing CAS in an older cached PubChem object when possible."""
        for identifier in (props.name, props.symbol, props.smiles):
            if not identifier:
                continue
            try:
                cid = self._get_pubchem_cid(identifier)
                if cid:
                    self._apply_pubchem_identity_properties(props, cid)
                    if props.CAS:
                        self._save_cache(cache_key, props.to_dict())
                        return
            except Exception:
                continue

    def _repair_cached_pubchem_sources(self, props: ChemicalProperties) -> None:
        """Normalize cached PubChem objects saved with old broad source labels."""
        match = re.match(r'^(local|online)\s+\((.+)\)$', props.source or '')
        if match:
            source_kind, source_detail = match.groups()
            if props.antoine_A is not None and not props.antoine_source:
                props.antoine_source = source_detail
                props.property_sources.setdefault(
                    'Antoine',
                    {
                        'source': source_kind,
                        'method': 'antoine',
                        'quality': 0.90 if source_kind == 'online' else 0.95,
                        'notes': f'Antoine coefficients from {source_detail}',
                    },
                )
            props.source = 'pubchem'

        warnings_text = '\n'.join(props.lookup_warnings or [])
        estimated_fields = {
            'Critical temperature': ('Tc', 'guldberg_rule', 0.55, 'Estimated from Tb'),
            'Critical pressure': ('Pc', 'atom_count_ring_tb_pc', 0.55, 'Estimated from formula, Tb, and structure/name'),
            'Acentric factor': ('omega', 'lee_kesler', 0.70, 'Estimated from Tb, Tc, and Pc'),
            'Heat of vaporization': ('Hvap', 'trouton', 0.45, 'Estimated from Tb'),
        }
        for warning_prefix, (attr, method, quality, notes) in estimated_fields.items():
            if warning_prefix in warnings_text:
                self._remember_existing_source(props, attr, 'estimated', method, quality, notes)
        if 'Boiling point and critical temperature' in warnings_text:
            self._remember_existing_source(
                props, 'Tb', 'estimated', 'mw_boiling_point', 0.35, 'Estimated from molecular weight'
            )
            self._remember_existing_source(
                props, 'Tc', 'estimated', 'guldberg_rule', 0.45,
                'Estimated from molecular-weight boiling-point estimate'
            )

    def fetch_by_cas(self, cas_number: str) -> Optional[ChemicalProperties]:
        """Fetch properties by CAS registry number"""
        # Clean CAS number
        cas_clean = cas_number.strip().replace(' ', '')
        cache_key = self._pubchem_component_cache_key(
            cas_clean,
            cas_lookup=True,
        )
        cached = self._get_cached(cache_key)
        if cached:
            if self._is_missing_cache(cached):
                return None
            return ChemicalProperties(**cached)
        
        # Try PubChem with CAS
        try:
            search_url = f"{self.PUBCHEM_BASE}/compound/name/{cas_clean}/cids/JSON"
            req = urllib.request.Request(search_url)
            req.add_header('User-Agent', 'PFD-Editor/1.0')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if 'IdentifierList' in data:
                    result = self.fetch_from_pubchem(cas_clean)
                    if result:
                        self._save_cache(cache_key, result.to_dict())
                    return result
        except Exception as e:
            if self._is_transient_lookup_error(e):
                return None
        
        self._save_missing_cache(cache_key)
        return None

    def _apply_pubchem_identity_properties(self, props: ChemicalProperties, cid: int):
        """Enrich component identity without duplicating property resolvers."""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'PFD-Editor/1.0')

            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
        except Exception:
            return

        for heading, text in self._iter_pubchem_information(data):
            heading_lower = heading.lower()

            if props.CAS == "" and heading_lower == 'cas':
                cas = self._parse_cas_number(text)
                if cas:
                    props.CAS = cas
                    props.property_sources.setdefault(
                        'CAS',
                        {
                            'source': 'pubchem',
                            'method': 'pubchem_pug_view',
                            'quality': 0.90,
                            'notes': 'CAS registry number from PubChem PUG-View',
                        },
                    )
                # Phase-change and critical scalars are intentionally resolved
                # by their dedicated multi-record systems.  This identity pass
                # must not create a competing first-value PubChem authority.
                return

    @staticmethod
    def _parse_cas_number(text: str) -> Optional[str]:
        match = re.search(r'\b(\d{2,7}-\d{2}-\d)\b', text)
        return match.group(1) if match else None

    def _iter_pubchem_information(self, node: dict, heading: str = ''):
        """Yield (heading, text) pairs from PubChem PUG-View records."""
        if not isinstance(node, dict):
            return

        current_heading = node.get('TOCHeading') or heading
        for item in node.get('Information', []):
            value = item.get('Value', {})
            texts = []

            for marked in value.get('StringWithMarkup', []):
                text = marked.get('String')
                if text:
                    texts.append(text)

            if 'Number' in value:
                unit = value.get('Unit', '')
                for number in value.get('Number', []):
                    texts.append(f"{number} {unit}".strip())

            if texts:
                yield current_heading, ' '.join(texts)

        for child in node.get('Section', []):
            yield from self._iter_pubchem_information(child, current_heading)

        record = node.get('Record')
        if isinstance(record, dict):
            yield from self._iter_pubchem_information(record, current_heading)

    def _looks_nonvolatile_ionic(self, props: ChemicalProperties) -> bool:
        """Heuristic for salts/inorganics where VLE estimates are inappropriate."""
        name = (props.name or '').lower()
        formula = props.formula or ''
        rules = self.lookup_rules.get('nonvolatile_ionic', {})
        name_tokens = [token.lower() for token in rules.get('name_tokens', [])]
        if any(token in name for token in name_tokens):
            return True

        metal_tokens = rules.get('formula_element_tokens', [])
        return any(re.search(rf'{token}(?:\\d|[A-Z]|$)', formula) for token in metal_tokens)

    @staticmethod
    def _property_field_status(props: ChemicalProperties) -> dict[str, bool]:
        return {
            'Tb': props.Tb is not None,
            'Hvap': props.Hvap is not None,
            'Tc': props.Tc is not None,
            'Pc': props.Pc is not None,
            'omega': props.omega is not None,
            'Antoine': props.antoine_A is not None,
        }

    def _build_lookup_warnings(
        self,
        identifier: str,
        props: ChemicalProperties,
        direct_fields: dict[str, bool],
    ) -> list[str]:
        """Describe missing or estimated properties after online lookup."""
        warnings = []
        still_missing = []

        final_fields = self._property_field_status(props)
        for field_name in ['Tb', 'Hvap', 'Tc', 'Pc', 'omega', 'Antoine']:
            if not final_fields.get(field_name, False):
                still_missing.append(field_name)

        if still_missing:
            warnings.append(
                f"Online lookup for '{identifier}' did not provide "
                f"{', '.join(still_missing)}. Some calculations may use lower-order "
                f"fallbacks or fail if those properties are required."
            )

        if props.antoine_A is None and props.Tb is not None and props.Hvap is not None:
            warnings.append(
                f"No Antoine coefficients found for '{identifier}'; vapor pressure "
                f"will use Clausius-Clapeyron from Tb and Hvap when needed."
            )

        if props.phase_at_STP == 'solid' and props.Tb is not None:
            warnings.append(
                f"'{identifier}' is solid at STP; liquid boiling/vaporization properties "
                f"are only appropriate above the melting point, and ambient vapor pressure "
                f"would require sublimation/fusion data."
            )

        return warnings

    @staticmethod
    def _dedupe_lookup_warnings(warnings: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for warning in warnings:
            if warning not in seen:
                deduped.append(warning)
                seen.add(warning)
        return deduped

    @staticmethod
    def _remember_existing_source(
        props: ChemicalProperties,
        attr: str,
        source: str,
        method: str,
        quality: float,
        notes: str = '',
    ) -> None:
        if getattr(props, attr, None) is None:
            return
        props.property_sources.setdefault(
            attr,
            {
                'source': source,
                'method': method,
                'quality': quality,
                'notes': notes,
            },
        )
    
    def _estimate_missing_properties(self, props: ChemicalProperties) -> ChemicalProperties:
        """
        Estimate missing properties using PropertyResolver and correlations.
        
        Resolution order for each property:
        1. Online database lookup (PubChem, NIST)
        2. Group contribution estimation
        3. Empirical correlations
        """
        inferred_phase = _inferred_phase_at_stp(props)
        if inferred_phase is not None:
            props.phase_at_STP = inferred_phase

        if self._looks_nonvolatile_ionic(props):
            props.lookup_warnings.append(
                f"'{props.symbol}' appears to be ionic or nonvolatile; volatile-liquid "
                f"properties were not estimated."
            )
            if not props.Cp_coeffs and props.MW > 0:
                props.Cp_coeffs = [4.0 * props.MW + 10, 0.01, 0, 0]
            return props

        # Try PropertyResolver first for comprehensive lookup
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            resolver = get_property_resolver()
            
            # Get Antoine coefficients if missing
            if props.antoine_A is None:
                antoine = self._get_resolver_antoine(resolver, props, local=True)
                if antoine:
                    props.antoine_A = antoine.A
                    props.antoine_B = antoine.B
                    props.antoine_C = antoine.C
                    props.antoine_Tmin = antoine.T_min
                    props.antoine_Tmax = antoine.T_max
                    props.antoine_source = antoine.source
                    props.property_sources.setdefault(
                        'Antoine',
                        {
                            'source': 'local',
                            'method': 'antoine',
                            'quality': 0.95,
                            'notes': f'Antoine coefficients from {antoine.source}',
                        },
                    )
                else:
                    # Try online lookup
                    antoine_online = self._get_resolver_antoine(resolver, props, local=False)
                    if antoine_online:
                        props.antoine_A = antoine_online.A
                        props.antoine_B = antoine_online.B
                        props.antoine_C = antoine_online.C
                        props.antoine_Tmin = antoine_online.T_min
                        props.antoine_Tmax = antoine_online.T_max
                        props.antoine_source = antoine_online.source
                        props.property_sources.setdefault(
                            'Antoine',
                            {
                                'source': 'online',
                                'method': 'antoine',
                                'quality': 0.90,
                                'notes': f'Antoine coefficients from {antoine_online.source}',
                            },
                        )
            
            # Get Hvap if missing
            if props.Hvap is None:
                hvap = resolver.get_hvap(props.symbol)
                if hvap:
                    props.Hvap = hvap
                else:
                    hvap_online = resolver.resolve_hvap(
                        props.symbol,
                        props.to_dict(),
                        allow_online=True,
                        allow_estimation=False,
                    )
                    if hvap_online and hvap_online.value is not None:
                        props.Hvap = hvap_online.value
                        props.property_sources['Hvap'] = {
                            'source': hvap_online.source,
                            'method': hvap_online.method,
                            'quality': hvap_online.quality,
                            'notes': hvap_online.notes,
                        }

            tb_result = resolver.resolve_boiling_point(
                props.symbol,
                props.to_dict(),
                allow_online=True,
                allow_estimation=True,
            )
            if tb_result and tb_result.value is not None:
                replace_tb = (
                    props.Tb is None
                    or str(tb_result.method).startswith('coolprop_')
                )
                if replace_tb:
                    props.Tb = tb_result.value
                    props.property_sources['Tb'] = {
                        'source': tb_result.source,
                        'method': tb_result.method,
                        'quality': tb_result.quality,
                        'notes': tb_result.notes,
                    }
            elif (
                tb_result
                and tb_result.method in {
                    'no_normal_boiling_point_at_1atm',
                    'invalid_normal_boiling_point_below_triple_point',
                }
                and (
                    (props.property_sources.get('Tb') or {}).get('method')
                    != 'pfd_component_override'
                )
            ):
                props.Tb = None
                props.property_sources.pop('Tb', None)

            triple = resolver.resolve_triple_point(
                props.symbol,
                props.to_dict(),
                allow_online=True,
            )
            for attr in ('Tt', 'Pt'):
                item = triple.get(attr)
                if not item or item.value is None:
                    continue
                replace_triple = (
                    getattr(props, attr) is None
                    or str(item.method).startswith('coolprop_')
                )
                if replace_triple:
                    setattr(props, attr, item.value)
                    props.property_sources[attr] = {
                        'source': item.source,
                        'method': item.method,
                        'quality': item.quality,
                        'notes': item.notes,
                    }

            constrained_tb = resolver.resolve_boiling_point(
                props.symbol,
                props.to_dict(),
                allow_online=True,
                allow_estimation=True,
            )
            if (
                constrained_tb.method in {
                    'no_normal_boiling_point_at_1atm',
                    'invalid_normal_boiling_point_below_triple_point',
                }
                and (
                    (props.property_sources.get('Tb') or {}).get('method')
                    != 'pfd_component_override'
                )
            ):
                props.Tb = None
                props.property_sources.pop('Tb', None)

            critical = resolver.resolve_critical_properties(
                props.symbol,
                props.to_dict(),
                allow_online=True,
                allow_estimation=True,
            )
            coolprop_critical = any(
                item
                and item.value is not None
                and (
                    str(item.method).startswith('coolprop_')
                    or item.method == 'coolprop_critical_identity'
                )
                for item in critical.values()
            )
            for attr in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                item = critical.get(attr)
                if not item or item.value is None:
                    continue
                replace_critical = getattr(props, attr) is None or (
                    coolprop_critical
                    and (
                        str(item.method).startswith('coolprop_')
                        or item.method == 'coolprop_critical_identity'
                    )
                ) or (
                    attr == 'Zc'
                    and item.method == 'critical_volume_identity'
                    and (
                        (props.property_sources.get(attr) or {}).get('method')
                        != 'pfd_component_override'
                    )
                )
                if replace_critical:
                    setattr(props, attr, item.value)
                    props.property_sources[attr] = {
                        'source': item.source,
                        'method': item.method,
                        'quality': item.quality,
                        'notes': item.notes,
                    }

        except ImportError:
            pass
        
        # Estimate Hvap if missing using Trouton's rule (enhanced)
        if props.Hvap is None and props.Tb is not None:
            # Trouton-Hildebrand-Everett rule for polar/nonpolar compounds
            if props.Tb < 250:  # Low boilers (gases)
                props.Hvap = 0.075 * props.Tb  # ~75 J/mol-K
            elif props.Tb > 400:  # High boilers
                props.Hvap = 0.095 * props.Tb  # ~95 J/mol-K
            else:
                props.Hvap = 0.088 * props.Tb  # ~88 J/mol-K (standard Trouton)
            self._remember_existing_source(
                props,
                'Hvap',
                'estimated',
                'trouton',
                0.45,
                'Estimated from Tb',
            )
            props.lookup_warnings.append(
                f"Heat of vaporization for '{props.symbol}' was estimated from Tb."
            )
        
        return props

    @staticmethod
    def _property_identifiers(
        props: ChemicalProperties,
        include_formula: bool = True,
    ) -> list[str]:
        identifiers = []
        values = [props.symbol, props.name]
        if include_formula:
            values.append(props.formula)
        values.append(props.CAS)
        for value in values:
            if value and value not in identifiers:
                identifiers.append(value)
        return identifiers

    def _get_resolver_antoine(
        self,
        resolver,
        props: ChemicalProperties,
        local: bool,
    ):
        for identifier in self._property_identifiers(props, include_formula=False):
            if local:
                antoine = resolver.get_antoine_local(identifier, props=props.to_dict())
            else:
                antoine = resolver.get_antoine_online(identifier, props=props.to_dict())
            if antoine:
                return antoine
        return None


class ChemicalDatabase:
    """Database of chemical properties with online fallback"""

    _SMILES_CACHE_SCHEMA_VERSION = 3
    
    def __init__(self, db_path: Optional[str] = None, enable_online: bool = True):
        """
        Initialize database from JSON file
        
        Args:
            db_path: Path to chemicals.json. If None, uses default location.
            enable_online: Whether to fetch from online sources for missing chemicals
        """
        self.chemicals: dict[str, ChemicalProperties] = {}
        self._aliases: dict[str, str] = {}
        self.enable_online = enable_online
        self.online_fetcher = OnlinePropertyFetcher() if enable_online else None
        self._hydrated_symbols: set[str] = set()
        
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "chemicals.json"
        self._smiles_cache_path = (
            Path(__file__).parent
            / "data"
            / "runtime"
            / "smiles_cache.sqlite"
        )
        
        self._load_database(db_path)
        self._build_aliases()
    
    def _load_database(self, path: str | Path):
        """Load chemicals from JSON file"""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Chemical database not found at {path}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for symbol, props in data.get('chemicals', {}).items():
            property_sources = self._expand_dataset_property_sources(props)
            self.chemicals[symbol] = ChemicalProperties(
                symbol=symbol,
                name=props.get('name', symbol),
                formula=props.get('formula', symbol),
                CAS=props.get('CAS', ''),
                smiles=props.get('smiles'),
                MW=props.get('MW', 0),
                Tc=props.get('Tc'),
                Pc=props.get('Pc'),
                Vc=props.get('Vc'),
                Zc=props.get('Zc'),
                omega=props.get('omega'),
                henry_Hcp=props.get('henry_Hcp'),
                henry_B=props.get('henry_B'),
                henry_Tmin=props.get('henry_Tmin'),
                henry_Tmax=props.get('henry_Tmax'),
                henry_Vinf=props.get('henry_Vinf'),
                henry_Vinf_uncertainty=props.get('henry_Vinf_uncertainty'),
                Tb=props.get('Tb'),
                Tt=props.get('Tt'),
                Pt=props.get('Pt'),
                Tm=props.get('Tm'),
                Hf=props.get('Hf'),
                Gf=props.get('Gf'),
                S=props.get('S'),
                Hf_liquid=props.get('Hf_liquid'),
                Gf_liquid=props.get('Gf_liquid'),
                S_liquid=props.get('S_liquid'),
                Hf_solid=props.get('Hf_solid'),
                Gf_solid=props.get('Gf_solid'),
                S_solid=props.get('S_solid'),
                Hcomb=props.get('Hcomb'),
                Hcomb_gross=props.get('Hcomb_gross'),
                Cp_coeffs=props.get('Cp_coeffs', []),
                Cp_liquid=props.get('Cp_liquid'),
                Cp_solid=props.get('Cp_solid'),
                rho_solid=props.get('rho_solid'),
                Vm_solid=props.get('Vm_solid'),
                dipole_moment=props.get('dipole_moment'),
                radius_of_gyration=props.get('radius_of_gyration'),
                modified_radius_of_gyration=props.get(
                    'modified_radius_of_gyration'
                ),
                hoc_eta=props.get('hoc_eta'),
                solid_material_form=props.get('solid_material_form', 'unspecified'),
                solid_polymorph=props.get('solid_polymorph', ''),
                property_correlations=props.get('property_correlations', {}),
                vapor_dimerization=props.get('vapor_dimerization'),
                uniquac_r=props.get('uniquac_r'),
                uniquac_q=props.get('uniquac_q'),
                Hvap=props.get('Hvap'),
                Hfus=props.get('Hfus'),
                fusion_transitions=list(props.get('fusion_transitions') or []),
                melting_transitions=list(props.get('melting_transitions') or []),
                Hsub=props.get('Hsub'),
                phase_at_STP=props.get('phase_at_STP', 'unknown'),
                critical_properties_unavailable=props.get('critical_properties_unavailable', False),
                lookup_warnings=props.get('lookup_warnings', []),
                property_sources=property_sources,
                antoine_A=props.get('antoine_A'),
                antoine_B=props.get('antoine_B'),
                antoine_C=props.get('antoine_C'),
                antoine_Tmin=props.get('antoine_Tmin'),
                antoine_Tmax=props.get('antoine_Tmax'),
                antoine_source=props.get('antoine_source'),
                source='local'
            )

    @staticmethod
    def _expand_dataset_property_sources(props: dict) -> dict:
        """Expand coarse dataset provenance into per-property source entries."""
        property_sources = dict(props.get('property_sources', {}) or {})
        dataset_source = property_sources.get('dataset')
        if not isinstance(dataset_source, dict):
            # Older chemicals.json records predate per-property provenance.
            # Their populated formation values are still curated local data;
            # retain that fact instead of presenting them as unknown-quality.
            for key in (
                'Hf', 'Gf', 'S',
                'Hf_liquid', 'Gf_liquid', 'S_liquid',
                'Hf_solid', 'Gf_solid', 'S_solid',
            ):
                if props.get(key) is None:
                    continue
                property_sources.setdefault(key, {
                    'source': 'local',
                    'method': 'chemicals_json',
                    'quality': 0.98,
                    'notes': 'Curated bundled chemicals.json formation property',
                })
            for key in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                if props.get(key) is None:
                    continue
                property_sources.setdefault(key, {
                    'source': 'local',
                    'method': 'chemicals_json',
                    'quality': 0.995,
                    'notes': 'Curated bundled chemicals.json critical property',
                })
            for key in ('Tm', 'Hfus'):
                if props.get(key) is None:
                    continue
                property_sources.setdefault(key, {
                    'source': 'local',
                    'method': 'chemicals_json',
                    'quality': 0.95,
                    'notes': (
                        'Bundled chemicals.json solid-liquid phase-change '
                        'property'
                    ),
                })
            return property_sources
        dataset_source = dict(dataset_source)
        dataset_source.setdefault('quality', 0.98)

        scalar_keys = (
            'MW',
            'Tc', 'Pc', 'Vc', 'Zc', 'omega',
            'Tb', 'Tt', 'Pt', 'Tm',
            'Hf', 'Gf', 'S',
            'Hf_liquid', 'Gf_liquid', 'S_liquid',
            'Hf_solid', 'Gf_solid', 'S_solid',
            'Hcomb', 'Hcomb_gross',
            'Hvap', 'Hfus', 'Hsub',
            'Cp_coeffs', 'Cp_liquid', 'Cp_solid', 'rho_solid', 'Vm_solid',
            'dipole_moment', 'radius_of_gyration',
            'modified_radius_of_gyration',
            'hoc_eta',
            'henry_Hcp', 'henry_B', 'henry_Tmin', 'henry_Tmax',
            'henry_Vinf', 'henry_Vinf_uncertainty',
            'uniquac_r', 'uniquac_q',
            'smiles',
        )
        antoine_keys = (
            'antoine_A', 'antoine_B', 'antoine_C',
            'antoine_Tmin', 'antoine_Tmax',
        )

        def source_for_key(key: str) -> dict:
            source = dict(dataset_source)
            if key in {'Cp_coeffs', 'Cp_liquid', 'Cp_solid'}:
                source['quality'] = 0.95
            elif key in {'Tm', 'Hfus'}:
                source['quality'] = 0.95
                source['notes'] = (
                    'Bundled chemicals.json solid-liquid phase-change '
                    'property without more specific per-property provenance'
                )
            return source

        for key in scalar_keys:
            value = props.get(key)
            if value is None or value == []:
                continue
            property_sources.setdefault(key, source_for_key(key))
        if all(props.get(key) is not None for key in ('antoine_A', 'antoine_B', 'antoine_C')):
            property_sources.setdefault('Antoine', dict(dataset_source))
            for key in antoine_keys:
                if props.get(key) is not None:
                    property_sources.setdefault(key, dict(dataset_source))

        property_sources.pop('dataset', None)
        return property_sources
    
    def _build_aliases(self):
        """Build alias mapping for common names, formulas, and SMILES"""
        self._aliases = {
            # Common names -> symbols
            'water': 'H2O',
            'hydrogen': 'H2',
            'nitrogen': 'N2',
            'oxygen': 'O2',
            'ammonia': 'NH3',
            'methane': 'CH4',
            'ethane': 'C2H6',
            'ethylene': 'C2H4',
            'propane': 'C3H8',
            'propylene': 'C3H6',
            'butane': 'C4H10',
            'n-butane': 'C4H10',
            'isobutane': 'iC4H10',
            'pentane': 'C5H12',
            'hexane': 'C6H14',
            'heptane': 'C7H16',
            'n-heptane': 'C7H16',
            'octane': 'C8H18',
            'benzene': 'C6H6',
            'toluene': 'C7H8',
            'methanol': 'CH3OH',
            'ethanol': 'C2H5OH',
            'acetone': 'CH3COCH3',
            'acetic acid': 'CH3COOH',
            'carbon dioxide': 'CO2',
            'carbon monoxide': 'CO',
            'hydrogen sulfide': 'H2S',
            'sulfur dioxide': 'SO2',
            'chlorine': 'Cl2',
            'argon': 'Ar',
            'helium': 'He',
            'ethylene oxide': 'C2H4O',
            # Additional common names
            'isopropanol': 'C3H7OH',
            '2-propanol': 'C3H7OH',
            'isopropyl alcohol': 'C3H7OH',
            '1-propanol': 'C3H8O',
            'n-propanol': 'C3H8O',
            'propanol': 'C3H8O',
            'formaldehyde': 'CH2O',
            'formic acid': 'HCOOH',
            'propionic acid': 'C3H6O2',
            'acetaldehyde': 'CH3CHO',
            'diethyl ether': '(C2H5)2O',
            'ether': '(C2H5)2O',
            'ethyl ether': '(C2H5)2O',
            'ethyl acetate': 'C4H8O2',
            'methyl acetate': 'C3H6O2',
            'chloroform': 'CHCl3',
            'dichloromethane': 'CH2Cl2',
            'methylene chloride': 'CH2Cl2',
            'carbon tetrachloride': 'CCl4',
            'carbon tet': 'CCl4',
            'cyclohexane': 'C6H12',
            'styrene': 'C8H8',
            'phenol': 'C6H5OH',
            'aniline': 'C6H7N',
            'nitrobenzene': 'C6H5NO2',
            'nitric oxide': 'NO',
            'nitrogen dioxide': 'NO2',
            'nitrous oxide': 'N2O',
            'hydrogen peroxide': 'H2O2',
            'hydrogen cyanide': 'HCN',
            'methyl chloride': 'CH3Cl',
            'vinyl chloride': 'C2H3Cl',
            # Alcohols
            'butanol': 'C4H9OH',
            '1-butanol': 'C4H9OH',
            'n-butanol': 'C4H9OH',
            'butan-1-ol': 'C4H9OH',
            # Other common chemicals
            'acetonitrile': 'CH3CN',
        }
        
        # SMILES -> symbol mappings
        self._smiles_aliases = {
            'O': 'H2O',
            'CO': 'CH3OH',      # Methanol SMILES
            'CCO': 'C2H5OH',    # Ethanol SMILES
            'CCCO': 'C3H8O',    # 1-Propanol
            'CC(C)O': 'C3H7OH', # Isopropanol
            'C': 'CH4',         # Methane
            'CC': 'C2H6',       # Ethane
            'C=C': 'C2H4',      # Ethylene
            'CCC': 'C3H8',      # Propane
            'CC=C': 'C3H6',     # Propylene
            'CCCC': 'C4H10',    # Butane
            'CCCCC': 'C5H12',   # Pentane
            'CCCCCC': 'C6H14',  # Hexane
            'CCCCCCC': 'C7H16', # Heptane
            'c1ccccc1': 'C6H6', # Benzene
            'Cc1ccccc1': 'C7H8', # Toluene
            'CC(=O)C': 'CH3COCH3', # Acetone
            'CC(=O)O': 'CH3COOH',  # Acetic acid
            'CC=O': 'CH3CHO',      # Acetaldehyde
            'C=O': 'CH2O',         # Formaldehyde
            'O=C=O': 'CO2',        # Carbon dioxide
            '[C-]#[O+]': 'CO',     # Carbon monoxide
            'N': 'NH3',            # Ammonia
            'N#N': 'N2',           # Nitrogen
            'O=O': 'O2',           # Oxygen
            '[H][H]': 'H2',        # Hydrogen
            'C1CCCCC1': 'C6H12',   # Cyclohexane
            'CCOCC': '(C2H5)2O',   # Diethyl ether
            'CC(=O)OCC': 'C4H8O2', # Ethyl acetate
            'ClC(Cl)Cl': 'CHCl3',  # Chloroform
            'ClCCl': 'CH2Cl2',     # Dichloromethane
            'ClC(Cl)(Cl)Cl': 'CCl4', # Carbon tetrachloride
        }
        
        formula_to_symbols = {}
        for symbol, props in self.chemicals.items():
            if props.formula:
                formula_to_symbols.setdefault(props.formula.lower(), set()).add(symbol)
            if props.CAS:
                self._aliases[props.CAS] = symbol
            if props.name:
                self._aliases[props.name.lower()] = symbol

        # Add formula aliases only when the local database has a unique match.
        for formula, symbols in formula_to_symbols.items():
            if len(symbols) == 1:
                self._aliases[formula] = next(iter(symbols))

    @staticmethod
    def _looks_like_formula(identifier: str) -> bool:
        text = str(identifier).strip()
        if not text or any(ch.isspace() for ch in text):
            return False
        return bool(re.fullmatch(r'(?:[A-Z][a-z]?\d*)+(?:[+-])?', text))

    @staticmethod
    def _resolver_props_dict(props: ChemicalProperties) -> dict:
        """Convert a ChemicalProperties object into resolver input."""
        return {
            'symbol': props.symbol,
            'name': props.name,
            'formula': props.formula,
            'CAS': props.CAS,
            'cas': props.CAS,
            'smiles': props.smiles,
            'MW': props.MW,
            'Tc': props.Tc,
            'Pc': props.Pc,
            'Vc': props.Vc,
            'Zc': props.Zc,
            'omega': props.omega,
            'henry_Hcp': props.henry_Hcp,
            'henry_B': props.henry_B,
            'henry_Tmin': props.henry_Tmin,
            'henry_Tmax': props.henry_Tmax,
            'henry_Vinf': props.henry_Vinf,
            'henry_Vinf_uncertainty': props.henry_Vinf_uncertainty,
            'Tb': props.Tb,
            'Tt': props.Tt,
            'Pt': props.Pt,
            'Tm': props.Tm,
            'Hf': props.Hf,
            'Gf': props.Gf,
            'S': props.S,
            'Hf_liquid': props.Hf_liquid,
            'Gf_liquid': props.Gf_liquid,
            'S_liquid': props.S_liquid,
            'Hf_solid': props.Hf_solid,
            'Gf_solid': props.Gf_solid,
            'S_solid': props.S_solid,
            'Hcomb': props.Hcomb,
            'Hcomb_gross': props.Hcomb_gross,
            'Cp_coeffs': props.Cp_coeffs,
            'Cp_liquid': props.Cp_liquid,
            'Cp_solid': props.Cp_solid,
            'rho_solid': props.rho_solid,
            'Vm_solid': props.Vm_solid,
            'dipole_moment': props.dipole_moment,
            'radius_of_gyration': props.radius_of_gyration,
            'modified_radius_of_gyration': props.modified_radius_of_gyration,
            'hoc_eta': props.hoc_eta,
            'solid_material_form': props.solid_material_form,
            'solid_polymorph': props.solid_polymorph,
            'property_correlations': props.property_correlations,
            'vapor_dimerization': props.vapor_dimerization,
            'property_sources': props.property_sources,
            'uniquac_r': props.uniquac_r,
            'uniquac_q': props.uniquac_q,
            'Hvap': props.Hvap,
            'Hfus': props.Hfus,
            'fusion_transitions': props.fusion_transitions,
            'melting_transitions': props.melting_transitions,
            'Hsub': props.Hsub,
            'critical_properties_unavailable': props.critical_properties_unavailable,
            'antoine_A': props.antoine_A,
            'antoine_B': props.antoine_B,
            'antoine_C': props.antoine_C,
            'antoine_Tmin': props.antoine_Tmin,
            'antoine_Tmax': props.antoine_Tmax,
            'antoine_source': props.antoine_source,
        }

    @staticmethod
    def _props_mapping(props: Optional[Any]) -> dict:
        if props is None:
            return {}
        if isinstance(props, ChemicalProperties):
            return props.to_dict()
        if isinstance(props, dict):
            return props
        to_dict = getattr(props, 'to_dict', None)
        if callable(to_dict):
            try:
                data = to_dict()
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _normalize_cas(value: Any) -> str:
        text = str(value or '').strip()
        if not text:
            return ''
        match = re.fullmatch(r'(\d{2,7})-?(\d{2})-?(\d)', text)
        if not match:
            return text
        return '-'.join(match.groups())

    @classmethod
    def _looks_like_cas(cls, value: Any) -> bool:
        return bool(re.fullmatch(r'\d{2,7}-?\d{2}-?\d', str(value or '').strip()))

    @staticmethod
    def _smiles_cache_key(identifier: Any) -> str:
        return re.sub(r'\s+', ' ', str(identifier or '').strip()).lower()

    def _ensure_smiles_cache(self) -> None:
        path = Path(self._smiles_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA busy_timeout = 30000")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == self._SMILES_CACHE_SCHEMA_VERSION:
                return
            conn.execute("BEGIN IMMEDIATE")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == self._SMILES_CACHE_SCHEMA_VERSION:
                return
            conn.execute("DROP TABLE IF EXISTS smiles_cache")
            conn.execute("DROP TABLE IF EXISTS opsin_negative_cache")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS smiles_cache (
                    identifier TEXT PRIMARY KEY,
                    smiles TEXT NOT NULL,
                    source TEXT NOT NULL,
                    quality REAL NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opsin_negative_cache (
                    identifier TEXT PRIMARY KEY,
                    opsin_version TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                f"PRAGMA user_version = {self._SMILES_CACHE_SCHEMA_VERSION}"
            )

    @staticmethod
    def _opsin_runtime_version() -> str:
        global _OPSIN_RUNTIME_VERSION
        if _OPSIN_RUNTIME_VERSION is None:
            try:
                from importlib import metadata
                _OPSIN_RUNTIME_VERSION = str(metadata.version('py2opsin'))
            except Exception:
                _OPSIN_RUNTIME_VERSION = ''
        return _OPSIN_RUNTIME_VERSION

    def _opsin_negative_cached(self, key: str) -> bool:
        memo = getattr(self, '_opsin_failure_memo', None)
        if memo is None:
            memo = set()
            self._opsin_failure_memo = memo
        if key in memo:
            return True
        self._ensure_smiles_cache()
        with sqlite3.connect(Path(self._smiles_cache_path)) as conn:
            row = conn.execute(
                """
                SELECT opsin_version FROM opsin_negative_cache
                WHERE identifier = ?
                """,
                (key,),
            ).fetchone()
        if row is None or row[0] != self._opsin_runtime_version():
            return False
        memo.add(key)
        return True

    def _record_opsin_negative(self, key: str, persist: bool) -> None:
        memo = getattr(self, '_opsin_failure_memo', None)
        if memo is None:
            memo = set()
            self._opsin_failure_memo = memo
        memo.add(key)
        if not persist:
            return
        self._ensure_smiles_cache()
        try:
            with sqlite3.connect(Path(self._smiles_cache_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO opsin_negative_cache (identifier, opsin_version)
                    VALUES (?, ?)
                    ON CONFLICT(identifier) DO UPDATE SET
                        opsin_version = excluded.opsin_version
                    """,
                    (key, self._opsin_runtime_version()),
                )
        except sqlite3.Error:
            pass

    def _cached_smiles(self, identifier: Any) -> Optional[SmilesResolution]:
        key = self._smiles_cache_key(identifier)
        if not key:
            return None
        self._ensure_smiles_cache()
        with sqlite3.connect(Path(self._smiles_cache_path)) as conn:
            row = conn.execute(
                """
                SELECT smiles, source, quality, notes
                FROM smiles_cache
                WHERE identifier = ?
                """,
                (key,),
            ).fetchone()
        if not row:
            return None
        smiles, source, quality, notes = row
        return SmilesResolution(
            smiles=smiles,
            source=source,
            method=f'{source}_smiles_cache',
            quality=float(quality),
            notes=notes or 'SMILES loaded from data/runtime/smiles_cache.sqlite',
            identifier=str(identifier),
        )

    def _cache_smiles(self, result: SmilesResolution, identifiers: list[str]) -> None:
        source = str(result.source or '').strip().lower()
        method = str(result.method or '').strip().lower()
        if (
            source in {'pfd', 'provided', 'user', 'override'}
            or 'pfd_' in method
            or 'override' in method
            or method == 'provided_smiles'
        ):
            return
        keys = {
            self._smiles_cache_key(identifier)
            for identifier in [result.identifier, *identifiers]
            if self._smiles_cache_key(identifier)
        }
        if not keys:
            return
        self._ensure_smiles_cache()
        with sqlite3.connect(Path(self._smiles_cache_path)) as conn:
            conn.executemany(
                """
                INSERT INTO smiles_cache
                    (identifier, smiles, source, quality, notes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(identifier) DO UPDATE SET
                    smiles = excluded.smiles,
                    source = excluded.source,
                    quality = excluded.quality,
                    notes = excluded.notes
                """,
                [
                    (
                        key,
                        result.smiles,
                        result.source,
                        float(result.quality),
                        result.notes or '',
                    )
                    for key in keys
                ],
            )

    @staticmethod
    def _smiles_from_props(props: Optional[Any]) -> Optional[SmilesResolution]:
        mapping = ChemicalDatabase._props_mapping(props)
        if not mapping:
            return None
        property_sources = mapping.get('property_sources') or {}
        for key in (
            'smiles', 'SMILES',
            'connectivity_smiles', 'ConnectivitySMILES',
            'canonical_smiles', 'CanonicalSMILES',
            'isomeric_smiles', 'IsomericSMILES',
        ):
            value = mapping.get(key)
            if not value:
                continue
            smiles = str(value).strip()
            if not smiles:
                continue
            source = property_sources.get(key) or property_sources.get('smiles') or {}
            return SmilesResolution(
                smiles=smiles,
                source=source.get('source') or 'provided',
                method=source.get('method') or 'provided_smiles',
                quality=float(source.get('quality', 1.0)),
                notes=source.get('notes') or 'SMILES supplied with component properties',
                identifier=str(mapping.get('name') or mapping.get('symbol') or ''),
            )
        return None

    @staticmethod
    def _smiles_from_inchi_identifier(identifier: str) -> Optional[SmilesResolution]:
        text = str(identifier or '').strip()
        if not text.lower().startswith('inchi='):
            return None
        try:
            from rdkit import Chem, rdBase
            with rdBase.BlockLogs():
                molecule = Chem.MolFromInchi(text)
            if molecule is None:
                return None
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        except Exception:
            return None
        if not smiles:
            return None
        return SmilesResolution(
            smiles=smiles,
            source='rdkit',
            method='rdkit_inchi_to_smiles',
            quality=1.0,
            notes='Molecular graph converted exactly from supplied InChI',
            identifier=text,
        )

    @staticmethod
    def _smiles_from_explicit_identifier(identifier: str) -> Optional[SmilesResolution]:
        text = str(identifier or '').strip()
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import classify_identifier
            else:
                from compound_identity import classify_identifier
            if classify_identifier(text) != 'smiles':
                return None
            from rdkit import Chem, rdBase
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                return None
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        except Exception:
            return None
        if not smiles:
            return None
        return SmilesResolution(
            smiles=smiles,
            source='provided',
            method='rdkit_smiles_identifier',
            quality=1.0,
            notes='Supplied SMILES validated and canonicalized by RDKit',
            identifier=text,
        )

    def _chemical_from_structure_resolution(
        self,
        identifier: str,
        structure: SmilesResolution,
    ) -> Optional[ChemicalProperties]:
        try:
            from rdkit import Chem, rdBase
            from rdkit.Chem import Descriptors, rdMolDescriptors
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(structure.smiles)
            if molecule is None:
                return None
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            molecular_weight = float(Descriptors.MolWt(molecule))
        except Exception:
            return None
        if not formula or not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
            return None

        props = ChemicalProperties(
            symbol=str(identifier),
            name=str(identifier),
            formula=formula,
            smiles=structure.smiles,
            MW=molecular_weight,
            phase_at_STP='unknown',
            source=f'{structure.source}_structure',
            lookup_warnings=[
                f"Identifier '{identifier}' resolved to molecular structure by "
                f"{structure.method}; physical properties are estimated unless "
                "a separate local or online source is selected."
            ],
            property_sources={
                'smiles': structure.property_source(),
                'formula': {
                    'source': 'calculated',
                    'method': 'rdkit_smiles_formula',
                    'quality': 0.99,
                    'notes': 'Molecular formula calculated from resolved structure',
                },
                'MW': {
                    'source': 'calculated',
                    'method': 'rdkit_smiles_molecular_weight',
                    'quality': 0.99,
                    'notes': 'Molecular weight calculated from resolved structure',
                },
            },
        )
        self._hydrate_properties(props, allow_online=False)
        return props

    def _smiles_from_local_database(self, identifier: str) -> Optional[SmilesResolution]:
        props = self.chemicals.get(identifier)
        if props is None:
            id_lower = str(identifier).lower()
            props = next((
                item for symbol, item in self.chemicals.items()
                if symbol.lower() == id_lower
            ), None)
        if props is None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                symbol = get_compound_identity_resolver().resolve_symbol(identifier)
                props = self.chemicals.get(symbol) if symbol else None
            except Exception:
                props = None
        if props is None:
            symbol = self._aliases.get(str(identifier).lower())
            props = self.chemicals.get(symbol) if symbol else None
        if props and props.smiles:
            source = (props.property_sources or {}).get('smiles') or {}
            return SmilesResolution(
                smiles=props.smiles,
                source=source.get('source') or 'local',
                method=source.get('method') or 'local_database_smiles',
                quality=float(source.get('quality', 0.98)),
                notes=source.get('notes') or f"SMILES from local ChemicalProperties for {props.name or props.symbol}",
                identifier=identifier,
            )

        if (
            not self._looks_like_formula(identifier)
            and identifier in getattr(self, '_smiles_aliases', {})
        ):
            symbol = self._smiles_aliases[identifier]
            aliased = self.chemicals.get(symbol)
            if aliased is not None:
                self._remember_resolved_smiles(aliased, identifier, 'local', 'local_smiles_alias')
            return SmilesResolution(
                smiles=identifier,
                source='local',
                method='local_smiles_alias',
                quality=1.0,
                notes=f"Identifier is a known local SMILES alias for {symbol}",
                identifier=identifier,
            )
        return None

    def _smiles_from_chemicals_metadata(
        self,
        identifier: str,
        expected_cas: str = '',
        expected_mw: Optional[float] = None,
    ) -> Optional[SmilesResolution]:
        try:
            from chemicals.identifiers import int_to_CAS, search_chemical
        except Exception:
            return None
        try:
            metadata = search_chemical(identifier)
        except Exception:
            return None
        smiles = str(getattr(metadata, 'smiles', '') or '').strip()
        if not smiles:
            return None
        resolved_cas = ''
        try:
            resolved_cas = int_to_CAS(int(metadata.CAS))
        except Exception:
            pass
        expected_cas = self._normalize_cas(expected_cas)
        if expected_cas and resolved_cas and resolved_cas != expected_cas:
            return None
        # Synonym search happily matches short component tags to unrelated
        # compounds ('ANI' -> 1-naphthyl isothiocyanate), so when the caller
        # knows the molecular weight, a badly disagreeing match is rejected.
        if expected_mw is not None:
            try:
                resolved_mw = float(getattr(metadata, 'MW', None))
            except (TypeError, ValueError):
                resolved_mw = None
            if (
                resolved_mw is not None
                and resolved_mw > 0.0
                and expected_mw > 0.0
                and abs(resolved_mw - expected_mw) > 0.02 * expected_mw
            ):
                return None
        name = getattr(metadata, 'common_name', None) or identifier
        return SmilesResolution(
            smiles=smiles,
            source='chemicals',
            method='chemicals_identifier_smiles',
            quality=0.99,
            notes=(
                f"SMILES resolved from local chemicals metadata for {name}"
                + (f" ({resolved_cas})" if resolved_cas else '')
            ),
            identifier=identifier,
        )

    def _smiles_from_opsin(self, identifier: str) -> Optional[SmilesResolution]:
        text = str(identifier or '').strip()
        if (
            not text
            or self._looks_like_formula(text)
            or self._looks_like_cas(text)
        ):
            return None
        cache_key = self._smiles_cache_key(text)
        if self._opsin_negative_cached(cache_key):
            return None
        try:
            from py2opsin import py2opsin
        except Exception:
            return None
        tmp_fpath = str(
            Path(tempfile.gettempdir())
            / f"pfdsim_py2opsin_{os.getpid()}_{uuid.uuid4().hex}.txt"
        )
        caught_warnings = []
        try:
            with warnings_module.catch_warnings(record=True) as records:
                warnings_module.simplefilter('always')
                smiles = py2opsin(text, output_format='SMILES', tmp_fpath=tmp_fpath)
                caught_warnings = list(records)
        except Exception:
            for warning in caught_warnings:
                warnings_module.warn(str(warning.message), warning.category, stacklevel=3)
            # The JVM may have died (rather than OPSIN rejecting the name),
            # so only remember the failure for this process.
            self._record_opsin_negative(cache_key, persist=False)
            return None
        for warning in caught_warnings:
            warnings_module.warn(str(warning.message), warning.category, stacklevel=3)
        if isinstance(smiles, list):
            smiles = smiles[0] if smiles else ''
        smiles = str(smiles or '').strip()
        if not smiles:
            # OPSIN ran and could not parse the name; deterministic, so the
            # negative is safe to persist (stamped with the OPSIN version).
            self._record_opsin_negative(cache_key, persist=True)
            return None
        warning_notes = '; '.join(str(warning.message) for warning in caught_warnings)
        ambiguous = any('ambigu' in str(warning.message).lower() for warning in caught_warnings)
        return SmilesResolution(
            smiles=smiles,
            source='opsin',
            method='py2opsin',
            quality=0.90 if ambiguous else 0.97,
            notes=(
                f"SMILES resolved from chemical name with OPSIN for {text}"
                + (f"; OPSIN warnings: {warning_notes}" if warning_notes else '')
            ),
            identifier=text,
        )

    def _smiles_from_pubchem(self, identifier: str) -> Optional[SmilesResolution]:
        if not self.enable_online or self.online_fetcher is None:
            return None
        try:
            cid = self.online_fetcher._get_pubchem_cid(identifier)
            if not cid:
                return None
            props_url = (
                f"{OnlinePropertyFetcher.PUBCHEM_BASE}/compound/cid/{cid}/property/"
                "MolecularFormula,MolecularWeight,IUPACName,"
                "ConnectivitySMILES,CanonicalSMILES,IsomericSMILES,InChI/JSON"
            )
            req = urllib.request.Request(props_url)
            req.add_header('User-Agent', 'PFD-Editor/1.0')
            with urllib.request.urlopen(req, timeout=10) as response:
                props_data = json.loads(response.read().decode())
            props = props_data['PropertyTable']['Properties'][0]
        except Exception:
            return None

        smiles = (
            props.get('ConnectivitySMILES')
            or props.get('CanonicalSMILES')
            or props.get('IsomericSMILES')
        )
        if not smiles:
            return None
        return SmilesResolution(
            smiles=str(smiles).strip(),
            source='pubchem',
            method='pubchem_structure',
            quality=0.97,
            notes=f"SMILES resolved through PubChem structure lookup for {identifier}",
            identifier=identifier,
            data=props,
        )

    @staticmethod
    def _remember_source(props: ChemicalProperties, attr: str, result) -> None:
        if not result or getattr(result, 'value', None) is None:
            return
        props.property_sources.setdefault(
            attr,
            {
                'source': getattr(result, 'source', ''),
                'method': getattr(result, 'method', ''),
                'quality': getattr(result, 'quality', None),
                'notes': getattr(result, 'notes', ''),
            },
        )

    @staticmethod
    def _remember_existing_source(
        props: ChemicalProperties,
        attr: str,
        source: str,
        method: str,
        quality: float,
        notes: str = '',
    ) -> None:
        if getattr(props, attr, None) is None:
            return
        props.property_sources.setdefault(
            attr,
            {
                'source': source,
                'method': method,
                'quality': quality,
                'notes': notes,
            },
        )

    def _set_missing_scalar(self, props: ChemicalProperties, attr: str, result) -> None:
        if not result or getattr(result, 'value', None) is None:
            return
        current = getattr(props, attr, None)
        source = props.property_sources.get(attr) or {}
        pfd_override = source.get('method') == 'pfd_component_override'
        coolprop_result = str(getattr(result, 'method', '') or '').startswith('coolprop_')
        independent_tm_validator = None
        if attr == 'Tm' and current is not None and coolprop_result:
            source_text = ' '.join(
                str(source.get(field) or '').strip().lower()
                for field in ('source', 'method', 'notes')
            )
            try:
                source_quality = float(source.get('quality', 1.0))
            except (TypeError, ValueError):
                source_quality = 0.0
            if (
                math.isfinite(source_quality)
                and source_quality >= 0.90
                and not str(source.get('method') or '').startswith('coolprop_')
                and not any(token in source_text for token in (
                    'estimated', 'estimate', 'provisional', 'nannoolal',
                    'joback', 'correlation fallback',
                ))
            ):
                independent_tm_validator = {
                    'value': current,
                    'source': source.get('source') or 'provided',
                    'method': source.get('method', 'direct'),
                    'quality': source_quality,
                    'notes': source.get('notes', ''),
                }
        if (
            current is not None
            and not (coolprop_result and not pfd_override)
            and not self._should_replace_scalar(props, attr, result)
        ):
            return
        setattr(props, attr, result.value)
        self._set_source(props, attr, result)
        if independent_tm_validator is not None:
            props.property_sources[attr][
                'independent_validator'
            ] = independent_tm_validator

    def _should_replace_scalar(self, props: ChemicalProperties, attr: str, result) -> bool:
        if getattr(props, attr, None) is None:
            return True
        if not self._stored_source_is_estimated(props, attr):
            return False
        result_source = str(getattr(result, 'source', '') or '').lower()
        result_method = str(getattr(result, 'method', '') or '').lower()
        if result_source == 'missing' or getattr(result, 'value', None) is None:
            return False
        result_is_estimated = result_source == 'estimated' or result_method in {
            'mw_correlation',
            'mw_boiling_point',
            'formula_hbd_boiling_point',
            'formula_no_hbd_boiling_point',
            'guldberg_rule',
            'lydersen_style',
            'atom_count_ring_tb_pc',
            'lee_kesler',
            'trouton',
        }
        if result_is_estimated:
            stored = props.property_sources.get(attr) or {}
            stored_quality = stored.get('quality')
            result_quality = getattr(result, 'quality', None)
            if stored_quality is None or result_quality is None:
                return False
            return float(result_quality) >= float(stored_quality)
        if result_method == 'direct' and result_source == 'provided':
            return False
        return True

    @staticmethod
    def _set_source(props: ChemicalProperties, attr: str, result) -> None:
        props.property_sources[attr] = {
            'source': getattr(result, 'source', ''),
            'method': getattr(result, 'method', ''),
            'quality': getattr(result, 'quality', None),
            'notes': getattr(result, 'notes', ''),
        }

    @staticmethod
    def _is_pfd_source(props: ChemicalProperties, attr: str) -> bool:
        source = props.property_sources.get(attr) or {}
        return source.get('method') == 'pfd_component_override'

    @staticmethod
    def _has_pfd_psat_override(props: ChemicalProperties) -> bool:
        correlations = props.property_correlations or {}
        correlation = correlations.get('Psat') or correlations.get('psat')
        return bool(
            isinstance(correlation, dict)
            and correlation.get('_pfd_override')
        )

    def _discard_stale_coupled_values_after_pfd_overrides(
        self,
        props: ChemicalProperties,
    ) -> None:
        if any(self._is_pfd_source(props, attr) for attr in ('Tc', 'Pc', 'Vc')):
            if not self._is_pfd_source(props, 'Zc'):
                props.Zc = None
                props.property_sources.pop('Zc', None)

        pfd_triple_fields = {
            attr for attr in ('Tt', 'Pt') if self._is_pfd_source(props, attr)
        }
        if len(pfd_triple_fields) == 1:
            counterpart = 'Pt' if 'Tt' in pfd_triple_fields else 'Tt'
            setattr(props, counterpart, None)
            props.property_sources.pop(counterpart, None)

        if self._has_pfd_psat_override(props):
            for attr in ('Tt', 'Pt', 'Tm'):
                if self._is_pfd_source(props, attr):
                    continue
                setattr(props, attr, None)
                props.property_sources.pop(attr, None)

    @staticmethod
    def _stored_source_is_estimated(props: ChemicalProperties, attr: str) -> bool:
        source = props.property_sources.get(attr) or {}
        if source.get('replaceable') is False:
            return False
        method = str(source.get('method') or '').lower()
        source_name = str(source.get('source') or '').lower()
        if source_name in {'missing', 'estimated'}:
            return True
        quality = source.get('quality')
        try:
            if quality is not None and float(quality) < 0.90:
                return True
        except (TypeError, ValueError):
            pass
        if source_name == 'exact':
            return False
        return method in {
            'lydersen_style',
            'atom_count_ring_tb_pc',
            'guldberg_rule',
            'mw_boiling_point',
            'mw_correlation',
            'formula_hbd_boiling_point',
            'formula_no_hbd_boiling_point',
            'atom_count_large_ring_vc',
            'liquid_gf_plus_standard_vaporization_gibbs',
            'trouton',
        }

    def _set_critical_scalar(self, props: ChemicalProperties, attr: str, result) -> None:
        if not result or getattr(result, 'value', None) is None:
            return
        current = getattr(props, attr, None)
        method = str(getattr(result, 'method', '') or '')
        source = props.property_sources.get(attr) or {}
        pfd_override = source.get('method') == 'pfd_component_override'
        replacement_locked = source.get('replaceable') is False
        coolprop_result = (
            method.startswith('coolprop_')
            or method == 'coolprop_critical_identity'
        )
        should_replace = (
            current is None
            or (coolprop_result and not pfd_override and not replacement_locked)
            or (
                attr == 'Zc'
                and method == 'critical_volume_identity'
                and not pfd_override
                and not replacement_locked
            )
            or (
                not replacement_locked
                and
                self._stored_source_is_estimated(props, attr)
                and self._should_replace_scalar(props, attr, result)
            )
        )
        if not should_replace:
            return
        setattr(props, attr, result.value)
        self._set_source(props, attr, result)

    def _refresh_estimated_dependents(self, props: ChemicalProperties) -> None:
        """Recompute stored estimates whose upstream scalar changed."""
        if self._stored_source_is_estimated(props, 'Hvap') and props.Tb is not None:
            if props.Tb < 250:
                hvap = 0.075 * props.Tb
            elif props.Tb > 400:
                hvap = 0.095 * props.Tb
            else:
                hvap = 0.088 * props.Tb
            props.Hvap = hvap
            props.property_sources['Hvap'] = {
                'source': 'estimated',
                'method': 'trouton',
                'quality': 0.55,
                'notes': 'Estimated from resolved Tb',
            }

    def _refresh_estimation_warnings(self, props: ChemicalProperties) -> None:
        def method_for(attr: str) -> str:
            source = props.property_sources.get(attr) or {}
            return str(source.get('method') or '').lower()

        stale_prefixes = (
            'Boiling point and critical temperature',
            'Boiling point for',
            'Critical temperature for',
            'Critical pressure for',
            'Acentric factor for',
            'Heat of vaporization for',
        )
        warnings = [
            warning for warning in (props.lookup_warnings or [])
            if not warning.startswith(stale_prefixes)
        ]
        if self._stored_source_is_estimated(props, 'Tb'):
            method = method_for('Tb')
            if method == 'nannoolal_tb':
                detail = 'Nannoolal group contribution'
            elif method in {'formula_hbd_boiling_point', 'formula_no_hbd_boiling_point'}:
                detail = 'formula atom-count fallback'
            elif method in {'mw_correlation', 'mw_boiling_point'}:
                detail = 'MW'
            else:
                detail = method or 'estimation'
            warnings.append(f"Boiling point for '{props.symbol}' was estimated from {detail}.")
        if self._stored_source_is_estimated(props, 'Tc'):
            method = method_for('Tc')
            if method == 'nannoolal_tc':
                detail = 'Nannoolal group contribution'
            elif method == 'guldberg_rule':
                detail = 'Tb using the Guldberg rule'
            else:
                detail = method or 'estimation'
            warnings.append(f"Critical temperature for '{props.symbol}' was estimated from {detail}.")
        if self._stored_source_is_estimated(props, 'Pc'):
            method = method_for('Pc')
            if method == 'nannoolal_pc':
                detail = 'Nannoolal group contribution'
            elif method == 'atom_count_ring_tb_pc':
                detail = 'formula, Tb, and structure/name'
            else:
                detail = method or 'estimation'
            warnings.append(f"Critical pressure for '{props.symbol}' was estimated from {detail}.")
        if self._stored_source_is_estimated(props, 'omega'):
            warnings.append(f"Acentric factor for '{props.symbol}' was estimated from Tb, Tc, and Pc.")
        if self._stored_source_is_estimated(props, 'Hvap'):
            warnings.append(f"Heat of vaporization for '{props.symbol}' was estimated from Tb.")
        props.lookup_warnings = OnlinePropertyFetcher._dedupe_lookup_warnings(warnings)

    def _hydrate_properties(self, props: ChemicalProperties, allow_online: bool = False) -> ChemicalProperties:
        """
        Fill missing scalar properties through the resolver.

        This pass is intentionally scalar-only. Temperature-dependent Perry
        correlations such as Cp, viscosity, density, and vapor pressure remain
        explicit resolver calls so their valid ranges and equation forms are
        preserved.
        """
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
        except ImportError:
            return props

        resolver = get_property_resolver()
        self._discard_stale_coupled_values_after_pfd_overrides(props)
        has_pfd_psat_override = self._has_pfd_psat_override(props)

        if not has_pfd_psat_override:
            melting = resolver.resolve_melting_point(
                props.symbol,
                self._resolver_props_dict(props),
                allow_online=allow_online,
            )
            self._set_missing_scalar(props, 'Tm', melting)

            triple = resolver.resolve_triple_point(
                props.symbol,
                self._resolver_props_dict(props),
                allow_online=allow_online,
            )
            for attr in ('Tt', 'Pt'):
                self._set_missing_scalar(props, attr, triple.get(attr))

        boiling = resolver.resolve_boiling_point(
            props.symbol,
            self._resolver_props_dict(props),
            allow_online=allow_online,
        )
        if (
            boiling
            and boiling.method in {
                'no_normal_boiling_point_at_1atm',
                'invalid_normal_boiling_point_below_triple_point',
            }
            and not self._is_pfd_source(props, 'Tb')
        ):
            props.Tb = None
            props.property_sources.pop('Tb', None)
        else:
            self._set_missing_scalar(props, 'Tb', boiling)

        critical = resolver.resolve_critical_properties(
            props.symbol,
            self._resolver_props_dict(props),
            allow_online=allow_online,
            allow_estimation=True,
        )
        for attr in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
            self._set_critical_scalar(props, attr, critical.get(attr))

        scalar_requests = (
            ('Hvap', lambda current: resolver.resolve_hvap(props.symbol, current, allow_online=allow_online, allow_estimation=allow_online)),
            ('Hfus', lambda current: resolver.resolve_hfus(props.symbol, current, allow_online=allow_online)),
        )
        for attr, resolve in scalar_requests:
            resolver_props = self._resolver_props_dict(props)
            result = resolve(resolver_props)
            if (
                attr == 'Hvap'
                and result
                and getattr(result, 'method', None) == 'nist_hvap_watson_fit'
                and getattr(result, 'value', None) is not None
            ):
                props.Hvap = result.value
                props.property_sources[attr] = {
                    'source': result.source,
                    'method': result.method,
                    'quality': result.quality,
                    'notes': result.notes,
                }
            else:
                self._set_missing_scalar(props, attr, result)

        fusion_route = getattr(resolver, 'resolve_fusion_transitions', None)
        if callable(fusion_route):
            props.fusion_transitions = [
                record.to_dict()
                for record in fusion_route(
                    props.symbol,
                    self._resolver_props_dict(props),
                    allow_online=allow_online,
                )
            ]
        melting_route = getattr(resolver, 'resolve_melting_transitions', None)
        if callable(melting_route):
            props.melting_transitions = [
                record.to_dict()
                for record in melting_route(
                    props.symbol,
                    self._resolver_props_dict(props),
                    allow_online=allow_online,
                )
            ]

        if str(props.phase_at_STP or '').strip().lower() in {'', 'unknown'}:
            inferred_phase = _inferred_phase_at_stp(props)
            if inferred_phase is not None:
                props.phase_at_STP = inferred_phase

        self._refresh_estimated_dependents(props)

        if allow_online and any(getattr(props, attr, None) is None for attr in ('Tc', 'Pc', 'Vc', 'Zc', 'omega')):
            resolver_props = self._resolver_props_dict(props)
            critical = resolver.resolve_critical_properties(
                props.symbol,
                resolver_props,
                allow_online=True,
                allow_estimation=True,
            )
            for attr in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                self._set_critical_scalar(props, attr, critical.get(attr))
            self._refresh_estimated_dependents(props)

        resolver_props = self._resolver_props_dict(props)
        formation = resolver.resolve_formation_properties(
            props.symbol,
            resolver_props,
            allow_online=allow_online,
        )
        for attr in ('Hf', 'Gf', 'S', 'Hcomb'):
            self._set_missing_scalar(props, attr, formation.get(attr))

        self._refresh_estimation_warnings(props)
        return props

    def _remember_resolved_smiles(self, props: Optional[ChemicalProperties],
                                  smiles: str, source: str,
                                  method: str = 'structure_resolution') -> None:
        if props is None:
            return
        if not props.smiles:
            props.smiles = smiles
        quality = 0.98
        if source.startswith('pubchem'):
            quality = 0.97
        elif source == 'opsin':
            quality = 0.97
        elif source in {'chemicals', 'local_chemicals'}:
            quality = 0.99
        props.property_sources.setdefault(
            'smiles',
            {
                'source': source,
                'method': method,
                'quality': quality,
                'notes': 'Resolved before structure fragmentation',
            },
        )

    def _structure_lookup_identifiers(self, identifier: str,
                                      props: Optional[Any]) -> list[str]:
        candidates: list[str] = []
        props_mapping = self._props_mapping(props)
        if isinstance(props, ChemicalProperties):
            for value in (props.CAS, props.name, props.symbol):
                if value and value not in candidates:
                    candidates.append(value)
            if props.formula and not self._looks_like_formula(props.formula):
                candidates.append(props.formula)
        elif props_mapping:
            for key in ('CAS', 'cas', 'name', 'symbol'):
                value = props_mapping.get(key)
                if value and value not in candidates:
                    candidates.append(str(value))
            formula = props_mapping.get('formula')
            if formula and not self._looks_like_formula(str(formula)):
                candidates.append(str(formula))
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            for value in get_compound_identity_resolver().candidate_identifiers(identifier):
                if value and value not in candidates:
                    candidates.append(value)
        except Exception:
            pass
        if identifier and identifier not in candidates:
            candidates.append(identifier)
        return candidates

    def _remember_smiles_for_local_matches(
        self,
        result: SmilesResolution,
        candidates: list[str],
    ) -> None:
        keys = {self._smiles_cache_key(candidate) for candidate in candidates if candidate}
        for props in self.chemicals.values():
            values = {
                self._smiles_cache_key(value)
                for value in (props.symbol, props.name, props.CAS)
                if value
            }
            if keys & values:
                self._remember_resolved_smiles(
                    props,
                    result.smiles,
                    result.source,
                    result.method,
                )

    def resolve_smiles_info(
        self,
        identifier: str,
        fetch_online: bool = True,
        props: Optional[Any] = None,
        candidates: Optional[list[str]] = None,
        expected_mw: Optional[float] = None,
    ) -> Optional[SmilesResolution]:
        """Resolve SMILES through the cache-first local -> chemicals -> OPSIN -> PubChem path."""
        props_mapping = self._props_mapping(props)
        expected_cas = self._normalize_cas(
            props_mapping.get('CAS') or props_mapping.get('cas') or ''
        )
        refrigerant_cas = None
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolution.coolprop import coolprop_cas_from_refrigerant_alias
            else:
                from property_resolution.coolprop import coolprop_cas_from_refrigerant_alias
            refrigerant_cas = coolprop_cas_from_refrigerant_alias(identifier)
        except Exception:
            pass

        cache_aliases: list[str] = []
        if refrigerant_cas and (not expected_cas or expected_cas == refrigerant_cas):
            # The exact CAS is the only safe provider/cache lookup key.  The
            # raw R-number is cached only after that identity has resolved.
            expected_cas = refrigerant_cas
            candidate_list = [refrigerant_cas]
            if identifier:
                cache_aliases.append(str(identifier))
        else:
            candidate_list = list(
                candidates or self._structure_lookup_identifiers(identifier, props)
            )
            if identifier:
                candidate_list.append(identifier)
            if refrigerant_cas and expected_cas != refrigerant_cas:
                ambiguous_key = self._smiles_cache_key(identifier)
                candidate_list = [
                    candidate for candidate in candidate_list
                    if self._smiles_cache_key(candidate) != ambiguous_key
                ]
        candidate_list = list(dict.fromkeys(
            str(candidate) for candidate in candidate_list if candidate
        ))
        cache_identifiers = list(dict.fromkeys([*candidate_list, *cache_aliases]))
        formula_only = self._looks_like_formula(identifier) and not props_mapping
        lookup_candidates = [
            candidate for candidate in candidate_list
            if not (formula_only and self._looks_like_formula(candidate))
        ]
        if expected_mw is None:
            try:
                expected_mw = float(props_mapping.get('MW') or props_mapping.get('mw'))
            except (TypeError, ValueError):
                expected_mw = None

        for candidate in candidate_list:
            cached = self._cached_smiles(candidate)
            if cached:
                if cache_aliases:
                    self._cache_smiles(cached, cache_identifiers)
                return cached

        explicit = self._smiles_from_props(props)
        if explicit:
            self._cache_smiles(explicit, cache_identifiers)
            return explicit

        direct_inchi = self._smiles_from_inchi_identifier(identifier)
        if direct_inchi:
            self._cache_smiles(direct_inchi, cache_identifiers)
            return direct_inchi

        direct_smiles = self._smiles_from_explicit_identifier(identifier)
        if direct_smiles:
            return direct_smiles

        for candidate in candidate_list:
            local = self._smiles_from_local_database(candidate)
            if local:
                self._cache_smiles(local, cache_identifiers)
                return local

        for candidate in lookup_candidates:
            chemicals_result = self._smiles_from_chemicals_metadata(
                candidate, expected_cas, expected_mw,
            )
            if chemicals_result:
                self._cache_smiles(chemicals_result, cache_identifiers)
                self._remember_smiles_for_local_matches(chemicals_result, candidate_list)
                return chemicals_result

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import classify_identifier
            else:
                from compound_identity import classify_identifier
            opsin_candidates = [
                candidate for candidate in lookup_candidates
                if classify_identifier(candidate) == 'name'
            ]
        except Exception:
            opsin_candidates = lookup_candidates

        for candidate in opsin_candidates:
            opsin_result = self._smiles_from_opsin(candidate)
            if opsin_result:
                self._cache_smiles(opsin_result, cache_identifiers)
                self._remember_smiles_for_local_matches(opsin_result, candidate_list)
                return opsin_result

        if not fetch_online or not self.enable_online or self.online_fetcher is None:
            return None

        for candidate in lookup_candidates:
            pubchem = self._smiles_from_pubchem(candidate)
            if pubchem:
                self._cache_smiles(pubchem, cache_identifiers)
                self._remember_smiles_for_local_matches(pubchem, candidate_list)
                return pubchem
        return None

    def _return_hydrated(self, symbol: str, props: ChemicalProperties) -> ChemicalProperties:
        """Hydrate a cached chemical once before handing it to callers."""
        key = symbol or props.symbol
        if key not in self._hydrated_symbols:
            self._hydrate_properties(props, allow_online=False)
            self._hydrated_symbols.add(key)
        return props
    
    def get(self, identifier: str, fetch_online: bool = True) -> Optional[ChemicalProperties]:
        """
        Get chemical properties by symbol, formula, name, SMILES, or CAS number
        
        Args:
            identifier: Chemical symbol (H2O), formula, common name, SMILES, or CAS number
            fetch_online: Whether to try online lookup if not in local database
            
        Returns:
            ChemicalProperties or None if not found
        """
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import (
                    classify_identifier,
                    get_compound_identity_resolver,
                )
            else:
                from compound_identity import (
                    classify_identifier,
                    get_compound_identity_resolver,
                )
            identifier_kind = classify_identifier(identifier)
            if (
                self._looks_like_formula(identifier)
                and get_compound_identity_resolver().is_ambiguous_formula(identifier)
            ):
                return None
        except Exception:
            identifier_kind = ''

        # Try direct match
        if identifier in self.chemicals:
            return self._return_hydrated(identifier, self.chemicals[identifier])
        
        # Try lowercase
        id_lower = identifier.lower()
        for symbol in self.chemicals:
            if symbol.lower() == id_lower:
                return self._return_hydrated(symbol, self.chemicals[symbol])

        # Try centralized identity resolver before local legacy aliases.
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            resolved = get_compound_identity_resolver().resolve_symbol(identifier)
            if resolved in self.chemicals:
                return self._return_hydrated(resolved, self.chemicals[resolved])
        except Exception:
            pass
        
        # Try alias (common name or formula)
        if id_lower in self._aliases:
            symbol = self._aliases[id_lower]
            if symbol in self.chemicals:
                return self._return_hydrated(symbol, self.chemicals[symbol])
        
        # Try SMILES alias (case-sensitive for SMILES)
        if identifier_kind == 'smiles' and identifier in self._smiles_aliases:
            symbol = self._smiles_aliases[identifier]
            if symbol in self.chemicals:
                return self._return_hydrated(symbol, self.chemicals[symbol])
        
        # Try CAS number match
        for symbol, props in self.chemicals.items():
            if props.CAS == identifier:
                return self._return_hydrated(symbol, props)

        # Bare formulas that are not uniquely present in chemicals.json should
        # not be guessed from non-CAS local tables; isomers can share formulas.
        if self._looks_like_formula(identifier):
            return None
        
        lookup_identifiers = [identifier]
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            lookup_identifiers = get_compound_identity_resolver().candidate_identifiers(identifier)
        except Exception:
            pass

        # Textbook and Perry are local databases. They stay available even when
        # network lookup is disabled.
        local_fetcher = self.online_fetcher or OnlinePropertyFetcher()
        for textbook_identifier in lookup_identifiers:
            textbook_result = local_fetcher.fetch_from_textbook(textbook_identifier)
            if textbook_result:
                self._hydrate_properties(textbook_result, allow_online=False)
                self.chemicals[textbook_result.symbol] = textbook_result
                self._hydrated_symbols.add(textbook_result.symbol)
                print(
                    f"Found '{textbook_result.name}' (MW={textbook_result.MW:.2f}) "
                    "from textbook"
                )
                return textbook_result

        local_formula = self._looks_like_formula(identifier)
        for perry_identifier in lookup_identifiers:
            if local_formula and perry_identifier != identifier:
                continue
            perry_result = local_fetcher.fetch_from_perry(perry_identifier)
            if perry_result:
                self._hydrate_properties(perry_result, allow_online=False)
                cache_symbol = perry_result.symbol
                if (
                    cache_symbol in self.chemicals
                    and not self._looks_like_formula(identifier)
                    and identifier != cache_symbol
                ):
                    cache_symbol = identifier
                    perry_result.symbol = identifier
                self.chemicals[cache_symbol] = perry_result
                self._hydrated_symbols.add(cache_symbol)
                print(
                    f"Found '{perry_result.name}' (MW={perry_result.MW:.2f}) "
                    "from Perry"
                )
                return perry_result

        structure = self.resolve_smiles_info(
            identifier,
            fetch_online=False,
        )
        if structure:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                local_symbol = (
                    get_compound_identity_resolver().resolve_structure_symbol(
                        structure.smiles
                    )
                )
            except Exception:
                local_symbol = None
            if local_symbol in self.chemicals:
                return self._return_hydrated(
                    local_symbol,
                    self.chemicals[local_symbol],
                )
            structure_result = self._chemical_from_structure_resolution(
                identifier,
                structure,
            )
            if structure_result:
                self.chemicals[structure_result.symbol] = structure_result
                self._hydrated_symbols.add(structure_result.symbol)
                return structure_result

        # Try online lookup if enabled
        if fetch_online and self.enable_online and self.online_fetcher:

            print(f"Chemical '{identifier}' not in local/textbook database, searching online...")
            
            # Check if it looks like a CAS number
            if re.match(r'^\d{2,7}-\d{2}-\d$', identifier):
                result = self.online_fetcher.fetch_by_cas(identifier)
            else:
                result = self.online_fetcher.fetch_from_pubchem(identifier)
            
            if result:
                # Add to local cache
                self._hydrate_properties(result, allow_online=True)
                self.online_fetcher.save_resolved_pubchem_cache(
                    identifier,
                    result,
                    cas_lookup=bool(re.match(r'^\d{2,7}-\d{2}-\d$', identifier)),
                )
                self.chemicals[result.symbol] = result
                self._hydrated_symbols.add(result.symbol)
                print(f"Found '{result.name}' (MW={result.MW:.2f}) from {result.source}")
                return result
            else:
                print(f"Chemical '{identifier}' not found in online databases")
        
        return None

    def get_user_component(self, identifier: str, fetch_online: bool = True) -> Optional[ChemicalProperties]:
        """Resolve a user-facing component identifier.

        Bare ambiguous formulas remain conservative in regular property lookup,
        but user component lists commonly use normal-alkane formulas such as
        C4H10. At that input boundary, interpret CnH2n+2 as the straight-chain
        local component when one exists.
        """
        # Simulator initialization registers resolved PFD aliases as exact
        # database keys before thermodynamics construction. Honor that explicit
        # process-local identity before applying conservative bare-formula
        # ambiguity rules in regular property lookup.
        if identifier in self.chemicals:
            return self._return_hydrated(identifier, self.chemicals[identifier])

        props = self.get(identifier, fetch_online=fetch_online)
        if props is not None:
            return props

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            symbol = get_compound_identity_resolver().straight_chain_alkane_symbol(identifier)
        except Exception:
            symbol = None
        if symbol and symbol in self.chemicals:
            return self._return_hydrated(symbol, self.chemicals[symbol])

        return None

    def search(self, query: str) -> list[ChemicalProperties]:
        """Search for chemicals matching query"""
        query = query.lower()
        results = []
        
        for symbol, props in self.chemicals.items():
            if (query in symbol.lower() or 
                query in props.name.lower() or
                query in props.formula.lower()):
                results.append(props)
        
        return results
    
    def list_all(self) -> list[str]:
        """Return list of all chemical symbols"""
        return list(self.chemicals.keys())
    
    def validate_components(self, component_symbols: list[str], 
                          fetch_missing: bool = True) -> tuple[list[str], list[str], list[str]]:
        """
        Validate that all components exist and have required properties
        
        Args:
            component_symbols: List of chemical symbols to validate
            fetch_missing: Whether to try fetching missing chemicals online
            
        Returns:
            (found, missing, fetched) - Lists of found, missing, and newly fetched symbols
        """
        found = []
        missing = []
        fetched = []
        
        for symbol in component_symbols:
            existing = symbol in self.chemicals
            props = self.get(symbol, fetch_online=fetch_missing)
            
            if props is not None:
                if existing:
                    found.append(symbol)
                else:
                    fetched.append(symbol)
            else:
                missing.append(symbol)
        
        return found, missing, fetched
    
    def get_MW(self, symbol: str) -> Optional[float]:
        """Get molecular weight for a species"""
        props = self.get(symbol)
        return props.MW if props else None
    
    def mixture_MW(self, composition: dict[str, float]) -> float:
        """Calculate mixture molecular weight [g/mol]"""
        mw = 0.0
        for symbol, x in composition.items():
            props = self.get(symbol)
            if props:
                mw += x * props.MW
        return mw
    
    def mixture_Cp(self, composition: dict[str, float], T: float) -> float:
        """Calculate ideal gas mixture heat capacity [J/mol-K]"""
        cp = 0.0
        for symbol, x in composition.items():
            props = self.get(symbol)
            if props:
                cp += x * props.Cp(T)
        return cp
    
    def add_chemical(self, props: ChemicalProperties):
        """Add or update a chemical in the database"""
        self.chemicals[props.symbol] = props
    
    def export_to_json(self, path: str):
        """Export database to JSON file"""
        data = {
            '_metadata': {
                'description': 'Chemical properties database',
                'units': {
                    'MW': 'g/mol',
                    'Tc': 'K',
                    'Pc': 'bar',
                    'Tb': 'K',
                    'Hf': 'kJ/mol'
                }
            },
            'chemicals': {
                symbol: props.to_dict()
                for symbol, props in self.chemicals.items()
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Global database instance
_db: Optional[ChemicalDatabase] = None


def get_database(enable_online: bool = True) -> ChemicalDatabase:
    """Get the global chemical database instance"""
    global _db
    if _db is None:
        _db = ChemicalDatabase(enable_online=enable_online)
    return _db


def get_chemical(symbol: str, fetch_online: bool = True) -> Optional[ChemicalProperties]:
    """Convenience function to get chemical properties"""
    return get_database().get(symbol, fetch_online=fetch_online)


def validate_components(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Convenience function to validate component list (returns found, missing)"""
    found, missing, fetched = get_database().validate_components(symbols)
    return found + fetched, missing


def search_chemicals(query: str) -> list[ChemicalProperties]:
    """Search for chemicals by name or formula"""
    return get_database().search(query)
