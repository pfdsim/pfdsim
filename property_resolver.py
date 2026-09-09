"""Compatibility facade for the property resolution system."""

from typing import Optional, Dict
import urllib.error
import urllib.parse
import urllib.request

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .property_resolution import (
        R, ONLINE_ANTOINE_TB_REL_TOL, ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
        CP_EXTRAPOLATION_LIMIT_K, LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K,
        MIN_EOS_PRESSURE_BAR, REFERENCE_TEMPERATURE_K, WATER_HVAP_298_KJ_PER_MOL,
        SOFT_PROPERTY_QUALITY_THRESHOLD, NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL,
        DensityObservation, FusionTransitionRecord, MeltingTransitionRecord,
        PropertyResolutionResult,
        AntoineCoefficients, HvapTemperatureFit,
        HeatCapacityLookup, HeatCapacityIntegralLookup, PropertyResolutionError,
        IdealGasCpKernel, KernelEvaluation, ChebyshevCpKernel,
        PolynomialCpKernel, ShomateCpKernel,
        LiquidCpKernel, ConstantLiquidCpKernel,
        LinearChebyshevLiquidCpKernel, NativeZabranskyLiquidCpKernel,
        PolynomialLiquidCpKernel, ScaledIdealGasLiquidCpKernel,
        ShomateLiquidCpKernel,
        SolidCpKernel, ConstantSolidCpKernel, PolynomialSolidCpKernel,
        ShomateSolidCpKernel, Perry151SolidCpKernel, TabularSolidCpKernel,
        PiecewiseSolidCpKernel, LastovkaSolidCpKernel,
        ModifiedKoppSolidCpKernel, SolidCpCollectionKernel,
        PropertyResolver,
    )
else:
    from property_resolution import (
        R, ONLINE_ANTOINE_TB_REL_TOL, ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
        CP_EXTRAPOLATION_LIMIT_K, LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K,
        MIN_EOS_PRESSURE_BAR, REFERENCE_TEMPERATURE_K, WATER_HVAP_298_KJ_PER_MOL,
        SOFT_PROPERTY_QUALITY_THRESHOLD, NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL,
        DensityObservation, FusionTransitionRecord, MeltingTransitionRecord,
        PropertyResolutionResult,
        AntoineCoefficients, HvapTemperatureFit,
        HeatCapacityLookup, HeatCapacityIntegralLookup, PropertyResolutionError,
        IdealGasCpKernel, KernelEvaluation, ChebyshevCpKernel,
        PolynomialCpKernel, ShomateCpKernel,
        LiquidCpKernel, ConstantLiquidCpKernel,
        LinearChebyshevLiquidCpKernel, NativeZabranskyLiquidCpKernel,
        PolynomialLiquidCpKernel, ScaledIdealGasLiquidCpKernel,
        ShomateLiquidCpKernel,
        SolidCpKernel, ConstantSolidCpKernel, PolynomialSolidCpKernel,
        ShomateSolidCpKernel, Perry151SolidCpKernel, TabularSolidCpKernel,
        PiecewiseSolidCpKernel, LastovkaSolidCpKernel,
        ModifiedKoppSolidCpKernel, SolidCpCollectionKernel,
        PropertyResolver,
    )

__all__ = (
    'urllib',
    'R',
    'ONLINE_ANTOINE_TB_REL_TOL',
    'ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K',
    'CP_EXTRAPOLATION_LIMIT_K',
    'LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K',
    'MIN_EOS_PRESSURE_BAR',
    'REFERENCE_TEMPERATURE_K',
    'WATER_HVAP_298_KJ_PER_MOL',
    'SOFT_PROPERTY_QUALITY_THRESHOLD',
    'NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL',
    'DensityObservation',
    'FusionTransitionRecord',
    'MeltingTransitionRecord',
    'PropertyResolutionResult',
    'AntoineCoefficients',
    'HvapTemperatureFit',
    'HeatCapacityLookup',
    'HeatCapacityIntegralLookup',
    'PropertyResolutionError',
    'IdealGasCpKernel',
    'KernelEvaluation',
    'ChebyshevCpKernel',
    'PolynomialCpKernel',
    'ShomateCpKernel',
    'LiquidCpKernel',
    'ConstantLiquidCpKernel',
    'LinearChebyshevLiquidCpKernel',
    'NativeZabranskyLiquidCpKernel',
    'PolynomialLiquidCpKernel',
    'ScaledIdealGasLiquidCpKernel',
    'ShomateLiquidCpKernel',
    'SolidCpKernel',
    'ConstantSolidCpKernel',
    'PolynomialSolidCpKernel',
    'ShomateSolidCpKernel',
    'Perry151SolidCpKernel',
    'TabularSolidCpKernel',
    'PiecewiseSolidCpKernel',
    'LastovkaSolidCpKernel',
    'ModifiedKoppSolidCpKernel',
    'SolidCpCollectionKernel',
    'PropertyResolver',
    'get_property_resolver',
    'resolve_vapor_pressure',
    'resolve_vapor_pressure_coefficients',
    'resolve_critical_properties',
    'resolve_boiling_point',
    'resolve_melting_point',
    'resolve_triple_point',
    'resolve_hvap',
    'resolve_hfus',
    'resolve_fusion_transitions',
    'resolve_melting_transitions',
    'resolve_heat_capacity',
    'resolve_ideal_gas_cp_kernel',
    'resolve_liquid_cp_kernel',
    'resolve_solid_cp_kernel',
    'resolve_liquid_molar_density',
    'resolve_density_observations',
    'resolve_solid_mass_density',
    'resolve_solid_molar_volume',
    'resolve_solid_molar_density',
    'resolve_liquid_molar_volume',
    'resolve_liquid_molar_volume_nearest',
    'resolve_viscosity',
    'resolve_surface_tension',
    'resolve_dipole_moment',
    'resolve_radius_of_gyration',
    'resolve_modified_radius_of_gyration',
    'resolve_radii_of_gyration',
    'resolve_formation_properties',
)

# Global resolver instance
_resolver: Optional[PropertyResolver] = None

def get_property_resolver() -> PropertyResolver:
    """Get global property resolver instance"""
    global _resolver
    if _resolver is None:
        _resolver = PropertyResolver()
    return _resolver


def resolve_vapor_pressure(symbol: str, T: float,
                           props: Dict = None,
                           allow_online: bool = True,
                           *,
                           minimum_pressure_bar: Optional[float] = None) -> PropertyResolutionResult:
    """Convenience function for vapor pressure resolution"""
    return get_property_resolver().resolve_vapor_pressure(
        symbol,
        T,
        props,
        allow_online=allow_online,
        minimum_pressure_bar=minimum_pressure_bar,
    )


def resolve_vapor_pressure_coefficients(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    *,
    minimum_pressure_bar: Optional[float] = None,
):
    """Return canonical Psat coefficients and continuation parameters."""
    return get_property_resolver().resolve_vapor_pressure_coefficients(
        symbol,
        props,
        allow_online=allow_online,
        minimum_pressure_bar=minimum_pressure_bar,
    )


def resolve_critical_properties(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    allow_estimation: bool = True,
) -> Dict[str, PropertyResolutionResult]:
    """Convenience function for critical properties resolution"""
    return get_property_resolver().resolve_critical_properties(
        symbol,
        props,
        allow_online=allow_online,
        allow_estimation=allow_estimation,
    )


def resolve_boiling_point(symbol: str, props: Dict = None) -> PropertyResolutionResult:
    """Convenience function for normal boiling point resolution."""
    return get_property_resolver().resolve_boiling_point(symbol, props)


def resolve_melting_point(symbol: str, props: Dict = None) -> PropertyResolutionResult:
    """Convenience function for normal melting point resolution."""
    return get_property_resolver().resolve_melting_point(symbol, props)


def resolve_triple_point(symbol: str, props: Dict = None) -> Dict[str, PropertyResolutionResult]:
    """Convenience function for triple-point temperature and pressure resolution."""
    return get_property_resolver().resolve_triple_point(symbol, props)


def resolve_hvap(
    symbol: str,
    props: Dict = None,
    T: Optional[float] = None,
) -> PropertyResolutionResult:
    """Convenience function for heat-of-vaporization resolution."""
    return get_property_resolver().resolve_hvap(symbol, props, T=T)


def resolve_hfus(symbol: str, props: Dict = None) -> PropertyResolutionResult:
    """Convenience function for heat-of-fusion resolution."""
    return get_property_resolver().resolve_hfus(symbol, props)


def resolve_fusion_transitions(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    material_form: Optional[str] = None,
) -> tuple[FusionTransitionRecord, ...]:
    """Return all known form-specific fusion-transition records."""
    return get_property_resolver().resolve_fusion_transitions(
        symbol,
        props,
        allow_online=allow_online,
        material_form=material_form,
    )


def resolve_melting_transitions(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    material_form: Optional[str] = None,
) -> tuple[MeltingTransitionRecord, ...]:
    """Return all known form-specific melting-transition records."""
    return get_property_resolver().resolve_melting_transitions(
        symbol,
        props,
        allow_online=allow_online,
        material_form=material_form,
    )


def resolve_heat_capacity(
    symbol: str,
    T: float,
    phase: str = 'liquid',
    props: Dict = None,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Convenience function for heat-capacity resolution."""
    return get_property_resolver().resolve_heat_capacity(
        symbol,
        T,
        phase,
        props,
        allow_online=allow_online,
    )


def resolve_ideal_gas_cp_kernel(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    allow_estimation: bool = True,
) -> Optional[IdealGasCpKernel]:
    """Resolve one reusable ideal-gas heat-capacity correlation."""
    return get_property_resolver().resolve_ideal_gas_cp_kernel(
        symbol,
        props,
        allow_online=allow_online,
        allow_estimation=allow_estimation,
    )


def resolve_liquid_cp_kernel(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    allow_estimation: bool = True,
) -> Optional[LiquidCpKernel]:
    """Resolve one reusable ordinary-liquid heat-capacity correlation."""
    return get_property_resolver().resolve_liquid_cp_kernel(
        symbol,
        props,
        allow_online=allow_online,
        allow_estimation=allow_estimation,
    )


def resolve_solid_cp_kernel(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
    allow_estimation: bool = True,
) -> Optional[SolidCpKernel]:
    """Resolve one reusable material-form-aware solid heat-capacity curve."""
    return get_property_resolver().resolve_solid_cp_kernel(
        symbol,
        props,
        allow_online=allow_online,
        allow_estimation=allow_estimation,
    )


def resolve_liquid_molar_density(
    symbol: str,
    T: float,
    props: Dict = None,
) -> PropertyResolutionResult:
    """Convenience function for liquid molar density resolution."""
    return get_property_resolver().resolve_liquid_molar_density(symbol, T, props)


def resolve_density_observations(
    symbol: str,
    props: Dict = None,
    *,
    allow_online: bool = True,
    phase: Optional[str] = None,
    material_form: Optional[str] = None,
) -> tuple[DensityObservation, ...]:
    """Return normalized, phase-classified intrinsic density reports."""
    return get_property_resolver().resolve_density_observations(
        symbol,
        props,
        allow_online=allow_online,
        phase=phase,
        material_form=material_form,
    )


def resolve_solid_mass_density(
    symbol: str,
    T: float,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve intrinsic solid mass density in kg/m^3."""
    return get_property_resolver().resolve_solid_mass_density(
        symbol,
        T,
        props,
        allow_online=allow_online,
    )


def resolve_solid_molar_volume(
    symbol: str,
    T: float,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve solid molar volume in m^3/kmol."""
    return get_property_resolver().resolve_solid_molar_volume(
        symbol,
        T,
        props,
        allow_online=allow_online,
    )


def resolve_solid_molar_density(
    symbol: str,
    T: float,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve solid molar density in mol/dm^3."""
    return get_property_resolver().resolve_solid_molar_density(
        symbol,
        T,
        props,
        allow_online=allow_online,
    )


def resolve_liquid_molar_volume(
    symbol: str,
    T: float,
    props: Dict = None,
) -> PropertyResolutionResult:
    """Convenience function for liquid molar volume resolution."""
    return get_property_resolver().resolve_liquid_molar_volume(symbol, T, props)


def resolve_liquid_molar_volume_nearest(
    symbol: str,
    T: float,
    props: Dict = None,
) -> PropertyResolutionResult:
    """Convenience function for liquid molar volume with endpoint clamping."""
    return get_property_resolver().resolve_liquid_molar_volume_nearest(symbol, T, props)


def resolve_viscosity(
    symbol: str,
    T: float,
    phase: str,
    props: Dict = None,
    *,
    P: Optional[float] = None,
    rho_molar: Optional[float] = None,
) -> PropertyResolutionResult:
    """Resolve viscosity; P is bar and rho_molar is kmol/m^3."""
    return get_property_resolver().resolve_viscosity(
        symbol,
        T,
        phase,
        props,
        P=P,
        rho_molar=rho_molar,
    )


def resolve_surface_tension(
    symbol: str,
    T: float,
    props: Dict = None,
) -> PropertyResolutionResult:
    """Convenience function for pure-component surface-tension resolution."""
    return get_property_resolver().resolve_surface_tension(symbol, T, props)


def resolve_dipole_moment(
    symbol: str,
    props: Dict = None,
    *,
    use_pvdz: bool = False,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve the permanent gas-phase molecular dipole in Debye."""
    return get_property_resolver().resolve_dipole_moment(
        symbol,
        props,
        use_pvdz=use_pvdz,
        allow_online=allow_online,
    )


def resolve_radius_of_gyration(
    symbol: str,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve the conventional mass-weighted radius of gyration [angstrom]."""
    return get_property_resolver().resolve_radius_of_gyration(
        symbol,
        props,
        allow_online=allow_online,
    )


def resolve_modified_radius_of_gyration(
    symbol: str,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> PropertyResolutionResult:
    """Resolve Thompson's modified radius of gyration ``R'`` [angstrom]."""
    return get_property_resolver().resolve_modified_radius_of_gyration(
        symbol,
        props,
        allow_online=allow_online,
    )


def resolve_radii_of_gyration(
    symbol: str,
    props: Dict = None,
    *,
    allow_online: bool = True,
) -> Dict[str, PropertyResolutionResult]:
    """Resolve both conventional and Thompson-modified gyration radii."""
    return get_property_resolver().resolve_radii_of_gyration(
        symbol,
        props,
        allow_online=allow_online,
    )


def resolve_formation_properties(
    symbol: str,
    props: Dict = None,
    allow_online: bool = True,
) -> Dict[str, PropertyResolutionResult]:
    """Convenience function for formation-property resolution."""
    return get_property_resolver().resolve_formation_properties(symbol, props, allow_online=allow_online)
