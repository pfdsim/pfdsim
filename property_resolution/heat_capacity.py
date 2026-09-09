# ruff: noqa: F401
# This module retains compatibility exports alongside HeatCapacityMixin.
from .common import *
import sqlite3
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .ideal_gas_cp import (
    DEFAULT_TMAX_K,
    DEFAULT_TMIN_K,
    GFN2_XTB_RRHO_QUALITY,
    AffineIdealGasCpKernel,
    ChebyshevCpKernel,
    IdealGasCpKernel,
    PiecewiseIdealGasCpKernel,
    PolynomialCpKernel,
    ShomateCpKernel,
    fit_chebyshev_kernel,
    fit_quality_penalty,
    kernel_from_payload,
    load_atom_increment_model,
    load_bundled_kernel,
    rrho_ideal_gas_heat_capacity,
    shifted_polynomial_coefficients,
)
from .liquid_cp import (
    ESTIMATOR_MAXIMUM_REDUCED_TEMPERATURE,
    ESTIMATOR_MINIMUM_REDUCED_TEMPERATURE,
    HBD_RATIO_GC_QUALITY_FACTOR,
    HBD_RATIO_MAXIMUM,
    HBD_RATIO_MINIMUM,
    MINIMUM_ESTIMATOR_CRITICAL_QUALITY,
    MIXED_DONOR_BONDI_QUALITY_FACTOR,
    ROWLINSON_BONDI_QUALITY_FACTOR,
    SCALED_IDEAL_GAS_QUALITY_FACTOR,
    STP_POINT_HALF_WIDTH_K,
    ConstantLiquidCpKernel,
    LinearChebyshevLiquidCpKernel,
    LiquidCpKernel,
    NativeZabranskyLiquidCpKernel,
    PolynomialLiquidCpKernel,
    ScaledIdealGasLiquidCpKernel,
    ShomateLiquidCpKernel,
    fit_linear_chebyshev_liquid_kernel,
    liquid_kernel_from_payload,
    load_bundled_liquid_kernel,
    lookup_bundled_liquid_cas,
)
from .solid_cp import (
    MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K,
    MODIFIED_KOPP_OTHER_J_MOL_K,
    SOLID_MINIMUM_TEMPERATURE_K,
    ConstantSolidCpKernel,
    LastovkaSolidCpKernel,
    ModifiedKoppSolidCpKernel,
    Perry151SolidCpKernel,
    PiecewiseSolidCpKernel,
    PolynomialSolidCpKernel,
    ShomateSolidCpKernel,
    SolidCpKernel,
    TabularSolidCpKernel,
    load_bundled_solid_kernel,
    lookup_bundled_solid_cas,
    solid_kernel_from_payload,
)
from .organic_classification import (
    HydrogenBondDonorProfile,
    classify_strict_molecular_organic,
    hydrogen_bond_donor_profile,
)
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..compound_identity import parse_formula_counts
else:
    from compound_identity import parse_formula_counts
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..solid_material_forms import normalize_solid_material_form
else:
    from solid_material_forms import normalize_solid_material_form
from .vapor_pressure import VaporPressureMixin


XTB_RRHO_ARTIFACT_VERSION = 1
XTB_RRHO_CACHE_NAMESPACE = 'qm_artifacts_v1'
XTB_RRHO_DERIVED_ORIGIN = 'xtb_rrho_v2'
XTB_RRHO_TMIN_K = DEFAULT_TMIN_K
XTB_RRHO_TMAX_K = DEFAULT_TMAX_K
XTB_RRHO_IMAGINARY_CUTOFF_CM_1 = 20.0
NIST_DIRECT_CP_ORIGIN = 'online_nist_direct_v2'
NIST_XTB_CP_ORIGIN = 'online_nist_xtb_policy_v1'
NIST_SHOMATE_CP_ORIGIN = 'online_nist_shomate_v2'
NIST_LEGACY_CP_ORIGIN = 'online_nist_legacy_v2'
COMPUTATIONAL_RRHO_QUALITY = 0.89


class HeatCapacityMixin:
        @staticmethod
        def _ideal_gas_phase(phase: str) -> bool:
            return phase.strip().lower().replace('-', '_') in {
                'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal',
            }


        @staticmethod
        def _ideal_gas_cp_identity(symbol: str, props: Dict[str, Any]) -> str:
            for value in (
                (props or {}).get('CAS'),
                (props or {}).get('cas'),
                symbol,
            ):
                text = str(value or '').strip()
                if re.fullmatch(r'\d{2,7}-\d{2}-\d', text):
                    return text
            return str(symbol or '').strip()


        @staticmethod
        def _ideal_gas_cp_props_fingerprint(props: Dict[str, Any]) -> str:
            payload = {
                'Cpg': ((props or {}).get('property_correlations') or {}).get('Cpg'),
                'Cp_coeffs': (props or {}).get('Cp_coeffs'),
                'Cp_coeffs_source': ((props or {}).get('property_sources') or {}).get('Cp_coeffs'),
                'CAS': (props or {}).get('CAS') or (props or {}).get('cas'),
                'formulas': tuple(
                    (props or {}).get(key)
                    for key in (
                        'formula', 'Formula', 'molecular_formula', 'MolecularFormula',
                    )
                ),
                'smiles': tuple(
                    (props or {}).get(key)
                    for key in (
                        'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                        'connectivity_smiles', 'ConnectivitySMILES',
                        'isomeric_smiles', 'IsomericSMILES',
                    )
                ),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


        def _provided_ideal_gas_cp_kernel(
            self,
            props: Dict[str, Any],
        ) -> Optional[IdealGasCpKernel]:
            correlations = (props or {}).get('property_correlations') or {}
            correlation = correlations.get('Cpg') if isinstance(correlations, dict) else None
            if isinstance(correlation, dict):
                equation = str(correlation.get('equation') or '').strip().lower()
                coefficients = self._correlation_coefficients(correlation)
                Tmin = float(correlation.get('Tmin_K', DEFAULT_TMIN_K))
                Tmax = float(correlation.get('Tmax_K', DEFAULT_TMAX_K))
                default_quality = 1.0 if correlation.get('_pfd_override') else 0.96
                try:
                    quality = float(correlation.get('quality', default_quality))
                except (TypeError, ValueError):
                    quality = default_quality
                source = str(correlation.get('source') or 'provided Cpg correlation')
                range_note = ''
                if correlation.get('Tmin_K') is None or correlation.get('Tmax_K') is None:
                    range_note = (
                        f'undeclared range defaulted to {DEFAULT_TMIN_K:g}-{DEFAULT_TMAX_K:g} K'
                    )
                fingerprint = hashlib.sha256(
                    json.dumps(correlation, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
                ).hexdigest()
                common = dict(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    quality=quality,
                    source=source,
                    method=f'provided_{equation}_ideal_gas_cp_kernel',
                    notes=range_note,
                    source_fingerprint=fingerprint,
                )
                if equation == 'shomate':
                    return ShomateCpKernel(
                        **common,
                        coefficients=tuple(coefficients.get(name, 0.0) for name in 'ABCDE'),
                    )
                if equation == 'poly_x':
                    shifted = shifted_polynomial_coefficients(
                        tuple(coefficients.get(name, 0.0) for name in 'ABCDEF')
                    )
                    return PolynomialCpKernel(**common, coefficients=shifted)
                if equation == 'exp_poly_x':
                    raw = tuple(coefficients.get(name, 0.0) for name in 'ABCDEF')

                    def evaluate(temperatures):
                        try:
                            import numpy as np
                            values = np.asarray(temperatures, dtype=float)
                            x = (values - 298.15) / 100.0
                            exponent = np.zeros_like(x)
                            for coefficient in reversed(raw):
                                exponent = exponent * x + coefficient
                            return np.exp(exponent)
                        except (ImportError, TypeError):
                            T = float(temperatures)
                            x = (T - 298.15) / 100.0
                            exponent = 0.0
                            for coefficient in reversed(raw):
                                exponent = exponent * x + coefficient
                            return math.exp(exponent)

                    return fit_chebyshev_kernel(
                        evaluate,
                        Tmin,
                        Tmax,
                        quality=quality,
                        source=source,
                        method='provided_exp_poly_x_chebyshev_ideal_gas_cp_kernel',
                        notes=range_note,
                        source_fingerprint=fingerprint,
                        degree=8,
                    )

            cp_coeffs = (props or {}).get('Cp_coeffs')
            if cp_coeffs and len(cp_coeffs) >= 4:
                source_info = ((props or {}).get('property_sources') or {}).get('Cp_coeffs') or {}
                try:
                    quality = (
                        1.0
                        if source_info.get('method') == 'pfd_component_override'
                        else float(source_info.get('quality', 0.95))
                    )
                except (TypeError, ValueError):
                    quality = 0.95
                return PolynomialCpKernel(
                    Tmin=DEFAULT_TMIN_K,
                    Tmax=DEFAULT_TMAX_K,
                    quality=quality,
                    source=str(source_info.get('source') or 'provided'),
                    method='provided_cubic_ideal_gas_cp_kernel',
                    notes=(
                        f'legacy Cp_coeffs range defaulted to '
                        f'{DEFAULT_TMIN_K:g}-{DEFAULT_TMAX_K:g} K'
                    ),
                    coefficients=tuple(float(value) for value in cp_coeffs[:4]),
                )
            return None


        @staticmethod
        def _liquid_cp_props_fingerprint(
            props: Dict[str, Any],
            provided_range: Optional[tuple[float, float, str]] = None,
        ) -> str:
            payload = {
                'Cpl': ((props or {}).get('property_correlations') or {}).get('Cpl'),
                'provided_Cpl_effective_range': provided_range,
                'Cp_liquid': (props or {}).get('Cp_liquid'),
                'Cp_liquid_source': ((props or {}).get('property_sources') or {}).get('Cp_liquid'),
                'CAS': (props or {}).get('CAS') or (props or {}).get('cas'),
                # The final 1.3x rung depends on the selected ideal-gas curve.
                'ideal_gas': HeatCapacityMixin._ideal_gas_cp_props_fingerprint(props),
                # Predictive liquid kernels depend on structure and critical
                # inputs even when no direct liquid-Cp value is present.
                'estimator_inputs': {
                    'Tc': (props or {}).get('Tc'),
                    'omega': (props or {}).get('omega'),
                    'formula': (props or {}).get('formula'),
                    'smiles': tuple(
                        (props or {}).get(key)
                        for key in (
                            'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                            'connectivity_smiles', 'ConnectivitySMILES',
                            'isomeric_smiles', 'IsomericSMILES',
                        )
                    ),
                    'sources': tuple(
                        ((props or {}).get('property_sources') or {}).get(key)
                        for key in ('Tc', 'omega', 'smiles')
                    ),
                },
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


        def _provided_liquid_cp_range(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> Optional[tuple[float, float, str]]:
            """Select explicit or phase-bounded limits for a provided Cpl curve."""
            correlations = (props or {}).get('property_correlations') or {}
            correlation = correlations.get('Cpl') if isinstance(correlations, dict) else None
            if not isinstance(correlation, dict):
                return None

            raw_Tmin = correlation.get('Tmin_K')
            raw_Tmax = correlation.get('Tmax_K')
            Tmin_missing = raw_Tmin is None
            Tmax_missing = raw_Tmax is None
            Tmin = float(raw_Tmin) if not Tmin_missing else DEFAULT_TMIN_K
            Tmax = float(raw_Tmax) if not Tmax_missing else DEFAULT_TMAX_K
            range_notes = {}

            if Tmin_missing:
                melting = self.resolve_melting_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if melting.value is not None:
                    try:
                        candidate = float(melting.value)
                    except (TypeError, ValueError, OverflowError):
                        candidate = math.nan
                    if math.isfinite(candidate) and candidate > 0.0:
                        Tmin = candidate
                        range_notes['Tmin'] = (
                            f'undeclared Tmin defaulted to resolved Tm={candidate:g} K '
                            f'({melting.source}/{melting.method})'
                        )
                if 'Tmin' not in range_notes:
                    range_notes['Tmin'] = (
                        f'undeclared Tmin defaulted to {DEFAULT_TMIN_K:g} K; '
                        'Tm unavailable'
                    )

            if Tmax_missing:
                boiling = self.resolve_boiling_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=allow_estimation,
                )
                if boiling.value is not None:
                    try:
                        candidate = float(boiling.value)
                    except (TypeError, ValueError, OverflowError):
                        candidate = math.nan
                    if math.isfinite(candidate) and candidate > 0.0:
                        Tmax = candidate
                        range_notes['Tmax'] = (
                            f'undeclared Tmax defaulted to resolved Tb={candidate:g} K '
                            f'({boiling.source}/{boiling.method})'
                        )
                if 'Tmax' not in range_notes:
                    range_notes['Tmax'] = (
                        f'undeclared Tmax defaulted to {DEFAULT_TMAX_K:g} K; '
                        'Tb unavailable'
                    )

            if Tmax <= Tmin:
                if Tmin_missing:
                    Tmin = DEFAULT_TMIN_K
                    range_notes['Tmin'] = (
                        f'undeclared Tmin defaulted to {DEFAULT_TMIN_K:g} K; '
                        'resolved Tm did not produce a valid liquid interval'
                    )
                if Tmax_missing:
                    Tmax = DEFAULT_TMAX_K
                    range_notes['Tmax'] = (
                        f'undeclared Tmax defaulted to {DEFAULT_TMAX_K:g} K; '
                        'resolved Tb did not produce a valid liquid interval'
                    )

            notes = '; '.join(
                range_notes[key] for key in ('Tmin', 'Tmax') if key in range_notes
            )
            return Tmin, Tmax, notes


        @staticmethod
        def _pfd_constant_liquid_cp(props: Dict[str, Any]) -> bool:
            source = ((props or {}).get('property_sources') or {}).get('Cp_liquid') or {}
            return source.get('method') == 'pfd_component_override'


        def _provided_liquid_cp_kernel(
            self,
            props: Dict[str, Any],
            provided_range: Optional[tuple[float, float, str]],
        ) -> Optional[LiquidCpKernel]:
            """Build an authoritative portable Cpl or explicit PFD constant."""
            correlations = (props or {}).get('property_correlations') or {}
            correlation = correlations.get('Cpl') if isinstance(correlations, dict) else None
            if isinstance(correlation, dict):
                equation = str(correlation.get('equation') or '').strip().lower()
                coefficients = self._correlation_coefficients(correlation)
                if provided_range is None:
                    raise ValueError('provided liquid Cp correlation requires a selected range')
                Tmin, Tmax, range_note = provided_range
                default_quality = 1.0 if correlation.get('_pfd_override') else 0.96
                try:
                    quality = float(correlation.get('quality', default_quality))
                except (TypeError, ValueError):
                    quality = default_quality
                source = str(correlation.get('source') or 'provided Cpl correlation')
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            'correlation': correlation,
                            'effective_range': provided_range,
                        },
                        sort_keys=True,
                        separators=(',', ':'),
                        default=str,
                    ).encode('utf-8')
                ).hexdigest()
                common = dict(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    quality=quality,
                    source=source,
                    method=f'provided_{equation}_liquid_cp_kernel',
                    notes=range_note,
                    source_fingerprint=fingerprint,
                )
                if equation == 'shomate':
                    return ShomateLiquidCpKernel(
                        **common,
                        coefficients=tuple(coefficients.get(name, 0.0) for name in 'ABCDE'),
                    )
                if equation == 'poly_x':
                    shifted = shifted_polynomial_coefficients(
                        tuple(coefficients.get(name, 0.0) for name in 'ABCDEF')
                    )
                    return PolynomialLiquidCpKernel(**common, coefficients=shifted)
                if equation == 'exp_poly_x':
                    raw = tuple(coefficients.get(name, 0.0) for name in 'ABCDEF')

                    def evaluate(temperatures):
                        try:
                            import numpy as np
                            values = np.asarray(temperatures, dtype=float)
                            x = (values - 298.15) / 100.0
                            exponent = np.zeros_like(x)
                            for coefficient in reversed(raw):
                                exponent = exponent * x + coefficient
                            return np.exp(exponent)
                        except (ImportError, TypeError):
                            T = float(temperatures)
                            x = (T - 298.15) / 100.0
                            exponent = 0.0
                            for coefficient in reversed(raw):
                                exponent = exponent * x + coefficient
                            return math.exp(exponent)

                    return fit_linear_chebyshev_liquid_kernel(
                        evaluate,
                        Tmin,
                        Tmax,
                        quality=quality,
                        source=source,
                        method='provided_exp_poly_x_chebyshev_liquid_cp_kernel',
                        notes=range_note,
                        source_fingerprint=fingerprint,
                    )

            if self._pfd_constant_liquid_cp(props) and props.get('Cp_liquid') is not None:
                source_info = ((props or {}).get('property_sources') or {}).get('Cp_liquid') or {}
                value = float(props['Cp_liquid'])
                fingerprint = hashlib.sha256(
                    f'pfd_constant_liquid_cp|{value:.17g}'.encode('utf-8')
                ).hexdigest()
                return ConstantLiquidCpKernel(
                    Tmin=1.0,
                    Tmax=10000.0,
                    quality=1.0,
                    source=str(source_info.get('source') or 'provided'),
                    method='provided_constant_liquid_cp_kernel',
                    notes='Explicit .pfd constant liquid heat capacity',
                    source_fingerprint=fingerprint,
                    value=value,
                    unbounded=True,
                )
            return None


        @staticmethod
        def _stored_liquid_cp_point_kernel(
            props: Dict[str, Any],
        ) -> Optional[ConstantLiquidCpKernel]:
            """Treat a non-PFD Cp_liquid scalar as one STP reference point."""
            if (props or {}).get('Cp_liquid') is None:
                return None
            if HeatCapacityMixin._pfd_constant_liquid_cp(props):
                return None
            source_info = ((props or {}).get('property_sources') or {}).get('Cp_liquid') or {}
            try:
                quality = float(source_info.get('quality', 0.95))
            except (TypeError, ValueError):
                quality = 0.95
            value = float(props['Cp_liquid'])
            fingerprint = hashlib.sha256(
                json.dumps(
                    {'value': value, 'source': source_info},
                    sort_keys=True,
                    separators=(',', ':'),
                    default=str,
                ).encode('utf-8')
            ).hexdigest()
            return ConstantLiquidCpKernel(
                Tmin=REFERENCE_TEMPERATURE_K - STP_POINT_HALF_WIDTH_K,
                Tmax=REFERENCE_TEMPERATURE_K + STP_POINT_HALF_WIDTH_K,
                quality=quality,
                source=str(source_info.get('source') or 'provided'),
                method='stp_point_constant_liquid_cp_kernel',
                notes=(
                    f'Cp_liquid interpreted as a single point at '
                    f'{REFERENCE_TEMPERATURE_K:g} K'
                ),
                source_fingerprint=fingerprint,
                value=value,
            )


        @staticmethod
        def _solid_cp_props_fingerprint(
            props: Dict[str, Any],
            provided_range: Optional[tuple[float, float, str]] = None,
        ) -> str:
            payload = {
                'Cps': ((props or {}).get('property_correlations') or {}).get('Cps'),
                'provided_Cps_effective_range': provided_range,
                'Cp_solid': (props or {}).get('Cp_solid'),
                'Cp_solid_source': ((props or {}).get('property_sources') or {}).get('Cp_solid'),
                'CAS': (props or {}).get('CAS') or (props or {}).get('cas'),
                'formula': (props or {}).get('formula'),
                'smiles': (props or {}).get('smiles'),
                'MW': (props or {}).get('MW'),
                'Tt': (props or {}).get('Tt'),
                'Tm': (props or {}).get('Tm'),
                'solid_material_form': (props or {}).get('solid_material_form'),
                'solid_polymorph': (props or {}).get('solid_polymorph'),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


        def _provided_solid_cp_range(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[tuple[float, float, str]]:
            """Select explicit or melting-bounded limits for a provided Cps curve."""
            correlations = (props or {}).get('property_correlations') or {}
            correlation = correlations.get('Cps') if isinstance(correlations, dict) else None
            if not isinstance(correlation, dict):
                return None

            raw_Tmin = correlation.get('Tmin_K')
            raw_Tmax = correlation.get('Tmax_K')
            Tmin_missing = raw_Tmin is None
            Tmax_missing = raw_Tmax is None
            Tmin = float(raw_Tmin) if not Tmin_missing else SOLID_MINIMUM_TEMPERATURE_K
            Tmax = float(raw_Tmax) if not Tmax_missing else DEFAULT_TMAX_K
            range_notes = {}

            melting = None
            if Tmin_missing or Tmax_missing:
                result = self.resolve_melting_point(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if result.value is not None:
                    try:
                        candidate = float(result.value)
                    except (TypeError, ValueError, OverflowError):
                        candidate = math.nan
                    if math.isfinite(candidate) and candidate > 0.0:
                        melting = candidate

            if Tmin_missing:
                Tmin = (
                    min(SOLID_MINIMUM_TEMPERATURE_K, 0.8 * melting)
                    if melting is not None
                    else SOLID_MINIMUM_TEMPERATURE_K
                )
                range_notes['Tmin'] = (
                    f'undeclared Tmin defaulted to {Tmin:g} K'
                    + (
                        f' from resolved Tm={melting:g} K'
                        if melting is not None and Tmin < SOLID_MINIMUM_TEMPERATURE_K
                        else ''
                    )
                )

            if Tmax_missing:
                if melting is not None:
                    Tmax = melting
                    range_notes['Tmax'] = (
                        f'undeclared Tmax defaulted to resolved Tm={melting:g} K '
                        f'({result.source}/{result.method})'
                    )
                else:
                    range_notes['Tmax'] = (
                        f'undeclared Tmax defaulted to {DEFAULT_TMAX_K:g} K; '
                        'Tm unavailable'
                    )

            notes = '; '.join(
                range_notes[key] for key in ('Tmin', 'Tmax') if key in range_notes
            )
            return Tmin, Tmax, notes


        @staticmethod
        def _pfd_constant_solid_cp(props: Dict[str, Any]) -> bool:
            source = ((props or {}).get('property_sources') or {}).get('Cp_solid') or {}
            return source.get('method') == 'pfd_component_override'


        @staticmethod
        def _solid_form_selection(props: Dict[str, Any]) -> tuple[str, str]:
            form = normalize_solid_material_form((props or {}).get('solid_material_form'))
            polymorph = str((props or {}).get('solid_polymorph') or '').strip()
            return form or 'unspecified', polymorph


        def _provided_solid_cp_kernel(
            self,
            props: Dict[str, Any],
            provided_range: Optional[tuple[float, float, str]],
        ) -> Optional[SolidCpKernel]:
            correlations = (props or {}).get('property_correlations') or {}
            correlation = correlations.get('Cps') if isinstance(correlations, dict) else None
            form, polymorph = self._solid_form_selection(props)
            if isinstance(correlation, dict):
                equation = str(correlation.get('equation') or '').strip().lower()
                coefficients = self._correlation_coefficients(correlation)
                if provided_range is None:
                    raise ValueError('provided solid Cp correlation requires a selected range')
                Tmin, Tmax, range_note = provided_range
                default_quality = 1.0 if correlation.get('_pfd_override') else 0.96
                try:
                    quality = float(correlation.get('quality', default_quality))
                except (TypeError, ValueError):
                    quality = default_quality
                source = str(correlation.get('source') or 'provided Cps correlation')
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            'correlation': correlation,
                            'effective_range': provided_range,
                        },
                        sort_keys=True,
                        separators=(',', ':'),
                        default=str,
                    ).encode()
                ).hexdigest()
                common = dict(
                    Tmin=Tmin, Tmax=Tmax, quality=quality, source=source,
                    method=f'provided_{equation}_solid_cp_kernel', notes=range_note,
                    source_fingerprint=fingerprint, material_form=form, polymorph=polymorph,
                    source_priority=0,
                )
                if equation == 'shomate':
                    return ShomateSolidCpKernel(
                        **common,
                        coefficients=tuple(coefficients.get(name, 0.0) for name in 'ABCDE'),
                    )
                if equation == 'poly_x':
                    return PolynomialSolidCpKernel(
                        **common,
                        coefficients=shifted_polynomial_coefficients(
                            tuple(coefficients.get(name, 0.0) for name in 'ABCDEF')
                        ),
                    )
                if equation == 'perry_151':
                    return Perry151SolidCpKernel(
                        **common,
                        coefficients=tuple(coefficients.get(name, 0.0) for name in 'ABCD'),
                    )

            if self._pfd_constant_solid_cp(props) and props.get('Cp_solid') is not None:
                source_info = ((props or {}).get('property_sources') or {}).get('Cp_solid') or {}
                value = float(props['Cp_solid'])
                return ConstantSolidCpKernel(
                    Tmin=1.0, Tmax=10000.0, quality=1.0,
                    source=str(source_info.get('source') or 'provided'),
                    method='provided_constant_solid_cp_kernel',
                    notes='Explicit .pfd constant solid heat capacity',
                    source_fingerprint=hashlib.sha256(
                        f'pfd_constant_solid_cp|{value:.17g}'.encode()
                    ).hexdigest(),
                    material_form=form, polymorph=polymorph, source_priority=0,
                    value=value, unbounded=True,
                )
            return None


        @staticmethod
        def _stored_solid_cp_point_kernel(props: Dict[str, Any]) -> Optional[ConstantSolidCpKernel]:
            if (props or {}).get('Cp_solid') is None or HeatCapacityMixin._pfd_constant_solid_cp(props):
                return None
            source_info = ((props or {}).get('property_sources') or {}).get('Cp_solid') or {}
            try:
                quality = float(source_info.get('quality', 0.95))
            except (TypeError, ValueError):
                quality = 0.95
            form, polymorph = HeatCapacityMixin._solid_form_selection(props)
            value = float(props['Cp_solid'])
            return ConstantSolidCpKernel(
                Tmin=REFERENCE_TEMPERATURE_K - STP_POINT_HALF_WIDTH_K,
                Tmax=REFERENCE_TEMPERATURE_K + STP_POINT_HALF_WIDTH_K,
                quality=quality, source=str(source_info.get('source') or 'provided'),
                method='stp_point_constant_solid_cp_kernel',
                notes=f'Cp_solid interpreted as a single point at {REFERENCE_TEMPERATURE_K:g} K',
                source_fingerprint=hashlib.sha256(
                    json.dumps({'value': value, 'source': source_info}, sort_keys=True, default=str).encode()
                ).hexdigest(),
                material_form=form, polymorph=polymorph, value=value,
            )


        @staticmethod
        def _ideal_gas_cp_formula_counts(formula: Any) -> Optional[dict[str, int]]:
            text = str(formula or '').strip()
            if not text:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
                counts = parse_formula_counts(text)
            except Exception:
                counts = None
            if not counts:
                try:
                    from chemicals.elements import simple_formula_parser
                    counts = simple_formula_parser(text)
                except Exception:
                    counts = None
            if not counts:
                return None
            normalized = {}
            for element, count in counts.items():
                try:
                    integer_count = int(count)
                except (TypeError, ValueError, OverflowError):
                    return None
                if integer_count <= 0 or float(count) != integer_count:
                    return None
                normalized[str(element)] = integer_count
            return normalized or None


        @staticmethod
        def _ideal_gas_cp_smiles_counts(smiles: Any) -> Optional[dict[str, int]]:
            text = str(smiles or '').strip()
            if not text:
                return None
            try:
                from rdkit import Chem
                molecule = Chem.MolFromSmiles(text)
                if molecule is None:
                    return None
                molecule = Chem.AddHs(molecule)
                counts: dict[str, int] = {}
                for atom in molecule.GetAtoms():
                    if atom.GetAtomicNum() <= 0:
                        return None
                    element = atom.GetSymbol()
                    counts[element] = counts.get(element, 0) + 1
                return counts or None
            except Exception:
                return None


        def _ideal_gas_cp_atom_counts(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[dict[str, int]]:
            for key in ('formula', 'Formula', 'molecular_formula', 'MolecularFormula'):
                formula = (props or {}).get(key)
                counts = self._ideal_gas_cp_formula_counts(formula)
                if counts:
                    return counts

            counts = self._ideal_gas_cp_formula_counts(symbol)
            if counts:
                return counts

            for key in (
                'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
                'isomeric_smiles', 'IsomericSMILES',
            ):
                smiles = (props or {}).get(key)
                counts = self._ideal_gas_cp_smiles_counts(smiles)
                if counts:
                    return counts

            resolved = self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            if resolved and resolved.value:
                counts = self._ideal_gas_cp_smiles_counts(resolved.value)
                if counts:
                    return counts
            return None


        @staticmethod
        def _xtb_rrho_dependency_state() -> dict[str, str]:
            dependencies = {}
            for name in ('tblite', 'ase', 'rdkit'):
                try:
                    dependencies[name] = version(name)
                except PackageNotFoundError:
                    dependencies[name] = 'missing'
            return dependencies


        def _xtb_rrho_artifact_cache(self):
            from .runtime_cache import SQLiteJSONCache

            path = self._runtime_json_cache().path
            state = getattr(self, '_xtb_rrho_artifact_cache_state', None)
            if state is not None and state[0] == path:
                return state[1]
            cache = SQLiteJSONCache(path, XTB_RRHO_CACHE_NAMESPACE)
            self._xtb_rrho_artifact_cache_state = path, cache
            return cache


        @staticmethod
        def _xtb_rrho_artifact_key(identity: str) -> str:
            digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
            return f'frequencies_v{XTB_RRHO_ARTIFACT_VERSION}_{digest}'


        def _load_xtb_rrho_artifact(
            self,
            identity: str,
        ) -> Optional[dict[str, Any]]:
            key = self._xtb_rrho_artifact_key(identity)
            try:
                payload = self._xtb_rrho_artifact_cache().get(key)
                if (
                    not isinstance(payload, dict)
                    or payload.get('version') != XTB_RRHO_ARTIFACT_VERSION
                    or payload.get('identity') != identity
                    or payload.get('method') != 'GFN2-xTB'
                ):
                    return None
                geometry = str(payload['geometry'])
                frequencies = tuple(float(value) for value in payload['frequencies_cm_1'])
                atom_count = int(payload['atom_count'])
                expected = (
                    0
                    if geometry == 'monatomic'
                    else 3 * atom_count - (5 if geometry == 'linear' else 6)
                )
                if (
                    geometry not in {'monatomic', 'linear', 'nonlinear'}
                    or atom_count <= 0
                    or len(frequencies) != expected
                    or any(not math.isfinite(value) or value <= 0.0 for value in frequencies)
                ):
                    raise ValueError('invalid cached GFN2-xTB RRHO artifact')
                return {
                    **payload,
                    'frequencies_cm_1': frequencies,
                    'atom_count': atom_count,
                }
            except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
                try:
                    self._xtb_rrho_artifact_cache().delete(key)
                except Exception:
                    pass
                return None


        def _save_xtb_rrho_artifact(
            self,
            identity: str,
            payload: dict[str, Any],
        ) -> None:
            record = {
                **payload,
                'version': XTB_RRHO_ARTIFACT_VERSION,
                'identity': identity,
                'method': 'GFN2-xTB',
            }
            try:
                self._xtb_rrho_artifact_cache().set(
                    self._xtb_rrho_artifact_key(identity),
                    record,
                )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        @staticmethod
        def _xtb_rrho_geometry(atoms) -> str:
            if len(atoms) == 1:
                return 'monatomic'
            if len(atoms) == 2:
                return 'linear'
            moments = np.sort(np.asarray(atoms.get_moments_of_inertia(), dtype=float))
            return 'linear' if moments[0] <= 1.0e-5 * moments[-1] else 'nonlinear'


        @classmethod
        def _calculate_xtb_rrho_artifact(
            cls,
            atoms,
            charge: int,
            multiplicity: int,
        ) -> dict[str, Any]:
            import tempfile

            from ase import units
            from ase.vibrations import Vibrations
            from tblite.ase import TBLite

            geometry = cls._xtb_rrho_geometry(atoms)
            if geometry == 'monatomic':
                energies = []
            else:
                atoms.calc = TBLite(
                    method='GFN2-xTB',
                    charge=charge,
                    multiplicity=multiplicity,
                    verbosity=0,
                )
                with tempfile.TemporaryDirectory(prefix='pfdsim-xtb-rrho-') as directory:
                    vibrations = Vibrations(
                        atoms,
                        name=str(Path(directory) / 'vib'),
                        delta=0.01,
                        nfree=2,
                    )
                    vibrations.run()
                    raw_energies = [complex(value) for value in vibrations.get_energies()]
                mode_count = 3 * len(atoms) - (5 if geometry == 'linear' else 6)
                energies = sorted(raw_energies, key=abs)[-mode_count:]

            cutoff_eV = XTB_RRHO_IMAGINARY_CUTOFF_CM_1 * units.invcm
            significant_imaginary = [
                energy for energy in energies if abs(energy.imag) > cutoff_eV
            ]
            if significant_imaginary:
                raise RuntimeError(
                    f'GFN2-xTB geometry has {len(significant_imaginary)} imaginary '
                    f'mode(s) above {XTB_RRHO_IMAGINARY_CUTOFF_CM_1:g} cm^-1'
                )
            frequencies = tuple(float(abs(energy) / units.invcm) for energy in energies)
            if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
                raise ValueError('GFN2-xTB produced invalid vibrational frequencies')
            return {
                'geometry': geometry,
                'atom_count': len(atoms),
                'frequencies_cm_1': frequencies,
                'imaginary_modes_below_cutoff': sum(bool(energy.imag) for energy in energies),
                'settings': {
                    'hessian': 'ASE central finite difference of tblite forces',
                    'displacement_angstrom': 0.01,
                    'imaginary_cutoff_cm_1': XTB_RRHO_IMAGINARY_CUTOFF_CM_1,
                },
            }


        def _xtb_rrho_ideal_gas_cp_kernel(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[IdealGasCpKernel]:
            dependencies = self._xtb_rrho_dependency_state()
            if dependencies.get('rdkit') == 'missing':
                return None

            smiles_result = self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            smiles = str(smiles_result.value or '').strip() if smiles_result else ''
            if not smiles:
                return None
            try:
                from rdkit import Chem

                molecule = self._validated_dipole_molecule(smiles)
                if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
                    return None
                canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
            except Exception:
                return None
            identity = f'smiles:{canonical_smiles}'
            cached_kernel = self._get_derived_cp_kernel(
                XTB_RRHO_DERIVED_ORIGIN,
                identity,
            )
            if cached_kernel is not None:
                return cached_kernel

            artifact = self._load_xtb_rrho_artifact(identity)
            if artifact is None and any(
                dependencies.get(name) == 'missing' for name in ('tblite', 'ase')
            ):
                return None

            signature = (
                f'artifact={XTB_RRHO_ARTIFACT_VERSION};'
                + ';'.join(
                    f'{name}={dependencies[name]}'
                    for name in ('tblite', 'ase', 'rdkit')
                )
            )
            attempts = getattr(self, '_xtb_rrho_attempt_cache', None)
            if attempts is None:
                attempts = {}
                self._xtb_rrho_attempt_cache = attempts
            attempt_key = (str(self.CACHE_DIR), identity)
            if (attempts.get(attempt_key) or {}).get('signature') == signature:
                return None

            try:
                if artifact is None:
                    atoms, charge, multiplicity = self._resolve_xtb_geometry(
                        identity,
                        canonical_smiles,
                        dependencies,
                    )
                    artifact = self._calculate_xtb_rrho_artifact(
                        atoms,
                        charge,
                        multiplicity,
                    )
                    artifact['dependencies'] = dependencies
                    self._save_xtb_rrho_artifact(identity, artifact)

                fingerprint = hashlib.sha256(
                    json.dumps(
                        artifact,
                        sort_keys=True,
                        separators=(',', ':'),
                        default=str,
                    ).encode('utf-8')
                ).hexdigest()
                evaluator = lambda temperatures: rrho_ideal_gas_heat_capacity(
                    temperatures,
                    artifact['frequencies_cm_1'],
                    artifact['geometry'],
                )
                kernel = None
                for degree in (8, 12):
                    candidate = fit_chebyshev_kernel(
                        evaluator,
                        XTB_RRHO_TMIN_K,
                        XTB_RRHO_TMAX_K,
                        quality=GFN2_XTB_RRHO_QUALITY,
                        source='calculated',
                        method='gfn2_xtb_rrho_ideal_gas_cp_kernel',
                        notes=(
                            'Gas-phase GFN2-xTB optimized-geometry numerical '
                            'frequencies with plain rigid-rotor/harmonic-oscillator '
                            'thermochemistry; fitted to a portable rational Chebyshev kernel'
                        ),
                        source_fingerprint=fingerprint,
                        degree=degree,
                    )
                    if (
                        candidate.fit_mape_percent < 0.01
                        and candidate.fit_max_error_percent < 0.1
                    ):
                        kernel = candidate
                        break
                if kernel is None:
                    raise ValueError('GFN2-xTB RRHO curve could not meet kernel fit tolerance')
                self._set_derived_cp_kernel(
                    XTB_RRHO_DERIVED_ORIGIN,
                    identity,
                    kernel,
                )
                attempts.pop(attempt_key, None)
                return kernel
            except Exception as exc:
                attempts[attempt_key] = {
                    'signature': signature,
                    'error': f'{type(exc).__name__}: {exc}',
                }
                return None


        def _atom_increment_ideal_gas_cp_kernel(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[IdealGasCpKernel]:
            model = load_atom_increment_model()
            if model is None:
                return None
            resolved = self._ideal_gas_cp_atom_counts(
                symbol,
                props,
                allow_online=allow_online,
            )
            if resolved is None:
                return None
            atom_counts = resolved
            cache_identity = model.cache_identity(atom_counts)
            if cache_identity is None:
                return None
            cached = self._get_derived_cp_kernel('atom_increment', cache_identity)
            if cached is not None:
                return cached
            kernel = model.kernel(atom_counts)
            if kernel is not None:
                self._set_derived_cp_kernel(
                    'atom_increment', cache_identity, kernel
                )
            return kernel


        @staticmethod
        def _derived_cp_cache_key(origin: str, identity: str) -> str:
            digest = hashlib.sha256(str(identity).strip().lower().encode('utf-8')).hexdigest()
            return f'ideal_gas_cp_{origin}_v1_{digest}'


        def _get_derived_cp_kernel(
            self,
            origin: str,
            identity: str,
        ) -> Optional[IdealGasCpKernel]:
            key = self._derived_cp_cache_key(origin, identity)
            try:
                payload = self._ideal_gas_cp_derived_cache().get(key)
                if payload and not self._is_missing_cache(payload):
                    return kernel_from_payload(payload)
            except (OSError, sqlite3.Error, TypeError, ValueError, KeyError):
                return None
            return None


        def _set_derived_cp_kernel(
            self,
            origin: str,
            identity: str,
            kernel: IdealGasCpKernel,
        ) -> None:
            key = self._derived_cp_cache_key(origin, identity)
            payload = kernel.to_payload()
            payload['origin'] = origin
            try:
                self._ideal_gas_cp_derived_cache().set(key, payload)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        @staticmethod
        def _derived_liquid_cp_cache_key(origin: str, identity: str) -> str:
            digest = hashlib.sha256(str(identity).strip().lower().encode('utf-8')).hexdigest()
            return f'liquid_cp_{origin}_v1_{digest}'


        def _get_derived_liquid_cp_kernel(
            self,
            origin: str,
            identity: str,
        ) -> Optional[LiquidCpKernel]:
            key = self._derived_liquid_cp_cache_key(origin, identity)
            try:
                payload = self._liquid_cp_derived_cache().get(key)
                if payload and not self._is_missing_cache(payload):
                    return liquid_kernel_from_payload(payload)
            except (OSError, sqlite3.Error, TypeError, ValueError, KeyError):
                return None
            return None


        def _set_derived_liquid_cp_kernel(
            self,
            origin: str,
            identity: str,
            kernel: LiquidCpKernel,
        ) -> None:
            key = self._derived_liquid_cp_cache_key(origin, identity)
            payload = kernel.to_payload()
            payload['origin'] = origin
            try:
                self._liquid_cp_derived_cache().set(key, payload)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        @staticmethod
        def _derived_solid_cp_cache_key(origin: str, identity: str) -> str:
            digest = hashlib.sha256(str(identity).strip().lower().encode('utf-8')).hexdigest()
            return f'solid_cp_{origin}_v1_{digest}'


        def _get_derived_solid_cp_kernel(
            self, origin: str, identity: str,
        ) -> Optional[SolidCpKernel]:
            key = self._derived_solid_cp_cache_key(origin, identity)
            try:
                payload = self._solid_cp_derived_cache().get(key)
                if payload and not self._is_missing_cache(payload):
                    return solid_kernel_from_payload(payload)
            except (OSError, sqlite3.Error, TypeError, ValueError, KeyError):
                return None
            return None


        def _set_derived_solid_cp_kernel(
            self, origin: str, identity: str, kernel: SolidCpKernel,
        ) -> None:
            key = self._derived_solid_cp_cache_key(origin, identity)
            payload = kernel.to_payload()
            payload['origin'] = origin
            try:
                self._solid_cp_derived_cache().set(key, payload)
            except (OSError, sqlite3.Error, TypeError, ValueError):
                pass


        @staticmethod
        def _nist_cp_cache_key(identifier: str) -> str:
            digest = hashlib.sha256(str(identifier).strip().lower().encode('utf-8')).hexdigest()
            # v3 adds phase-preserving solid tables and solid Shomate ranges.
            return f'cp_nist_v3_{digest}'


        def _fetch_nist_cp_source(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_network: bool,
        ) -> Optional[Dict[str, Any]]:
            """Return cached or newly fetched normalized NIST Cp source data."""
            candidates = self._identifier_candidates(symbol, props)
            cas = (props or {}).get('CAS') or (props or {}).get('cas')
            if cas:
                candidates = [str(cas)] + [item for item in candidates if item != str(cas)]

            uncached = []
            for index, candidate in enumerate(candidates):
                cache_key = self._nist_cp_cache_key(candidate)
                cached = self._get_cache(cache_key)
                if cached is not None:
                    if self._is_missing_cache(cached):
                        # A CAS query is authoritative for the requested
                        # identity.  Do not repeat weaker name/formula queries
                        # while its 90-day negative is fresh.
                        if cas and index == 0 and candidate == str(cas):
                            self._record_online_attempt_state(
                                OnlineAttemptState.COMPLETE_NO_DATA
                            )
                            return None
                        continue
                    return cached
                uncached.append((candidate, cache_key))

            if not allow_network:
                self._record_online_attempt_state(OnlineAttemptState.NOT_ATTEMPTED)
                return None

            transient_failure = False
            for candidate, cache_key in uncached:
                query_specs = []
                if re.fullmatch(r'\d{2,7}-\d{2}-\d', candidate):
                    query_specs.append({'ID': 'C' + candidate.replace('-', '')})
                query_specs.extend(({'Name': candidate}, {'Formula': candidate}))
                complete_no_data = True
                for query in query_specs:
                    try:
                        params = urllib.parse.urlencode({**query, 'Units': 'SI', 'Mask': '3'})
                        req = urllib.request.Request(f"{self.NIST_WEBBOOK}?{params}")
                        req.add_header('User-Agent', 'PFD-Editor/1.0')
                        with urllib.request.urlopen(req, timeout=15) as response:
                            html = response.read().decode('utf-8', errors='ignore')
                        tables = self._parse_nist_cp_tables(html)
                        if tables:
                            tables['_source'] = 'NIST Chemistry WebBook'
                            tables['_identifier'] = candidate
                            self._set_cache(cache_key, tables)
                            self._record_online_attempt_state(OnlineAttemptState.COMPLETE_WITH_DATA)
                            return tables
                    except Exception as error:
                        if self._is_transient_lookup_error(error):
                            transient_failure = True
                            complete_no_data = False
                if complete_no_data:
                    self._set_missing_cache(cache_key)

            if transient_failure:
                self._record_online_attempt_state(OnlineAttemptState.TRANSIENT_FAILURE)
                return None
            self._record_online_attempt_state(OnlineAttemptState.COMPLETE_NO_DATA)
            return None


        def _kernel_from_nist_source(
            self,
            tables: Dict[str, Any],
            *,
            native_only: bool = False,
        ) -> Optional[IdealGasCpKernel]:
            fingerprint = hashlib.sha256(
                json.dumps(tables, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
            ).hexdigest()
            segments = tables.get('gas_shomate') or []
            if segments:
                normalized = []
                for segment in segments:
                    try:
                        normalized.append(ShomateCpKernel(
                            Tmin=float(segment['Tmin_K']),
                            Tmax=float(segment['Tmax_K']),
                            quality=0.975,
                            source='NIST Chemistry WebBook',
                            method='nist_native_shomate_ideal_gas_cp_kernel',
                            notes='Native online NIST Shomate equation',
                            source_fingerprint=fingerprint,
                            coefficients=tuple(float(segment[name]) for name in 'ABCDE'),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
                if len(normalized) == 1:
                    return normalized[0]
                if normalized:
                    Tmin = min(kernel.Tmin for kernel in normalized)
                    Tmax = max(kernel.Tmax for kernel in normalized)

                    def evaluate(temperatures):
                        import numpy as np
                        values_T = np.asarray(temperatures, dtype=float)
                        result = np.full(values_T.shape, np.nan, dtype=float)
                        for kernel in normalized:
                            mask = (
                                np.isnan(result)
                                & (values_T >= kernel.Tmin - 1.0e-9)
                                & (values_T <= kernel.Tmax + 1.0e-9)
                            )
                            if np.any(mask):
                                result[mask] = [kernel._cp_native(float(T)) for T in values_T[mask]]
                        if np.any(np.isnan(result)):
                            valid = np.flatnonzero(~np.isnan(result))
                            if len(valid) < 2:
                                return result
                            result = np.interp(values_T, values_T[valid], result[valid])
                        return result

                    return fit_chebyshev_kernel(
                        evaluate,
                        Tmin,
                        Tmax,
                        quality=0.975,
                        source='NIST Chemistry WebBook',
                        method='nist_multirange_shomate_chebyshev_ideal_gas_cp_kernel',
                        notes=f'Collapsed {len(normalized)} native NIST Shomate ranges to one curve',
                        source_fingerprint=fingerprint,
                        degree=8,
                    )

            if native_only:
                return None

            points = self._nist_gas_cp_points(tables)
            if not points:
                return None
            Tmin = points[0][0]
            Tmax = points[-1][0]
            if len(points) == 1:
                point_T = points[0][0]
                Tmin = max(1.0e-9, point_T - 5.0)
                Tmax = point_T + 5.0
                return ShomateCpKernel(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    quality=0.84,
                    source='NIST Chemistry WebBook',
                    method='nist_constant_ideal_gas_cp_kernel',
                    notes=(
                        f'Single tabulated Cp point at {point_T:g} K; '
                        f'native constant range {Tmin:g}-{Tmax:g} K'
                    ),
                    source_fingerprint=fingerprint,
                    coefficients=(points[0][1], 0.0, 0.0, 0.0, 0.0),
                )
            if len(points) < 10:
                linear = self._linear_cp_fit(points)
                if linear is None:
                    return None
                intercept, slope, mape, maximum = linear
                sparse_penalty = (
                    0.01 * max(0.0, maximum - 2.0)
                    + 0.06 * max(0.0, mape - 0.5)
                )
                return ShomateCpKernel(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    quality=max(0.0, 0.86 - sparse_penalty),
                    source='NIST Chemistry WebBook',
                    method='nist_linear_ideal_gas_cp_kernel',
                    notes=f'Linear fit to {len(points)} tabulated Cp points',
                    fit_mape_percent=mape,
                    fit_max_error_percent=maximum,
                    source_fingerprint=fingerprint,
                    coefficients=(intercept, slope * 1000.0, 0.0, 0.0, 0.0),
                )
            fit = self._fit_shomate_cp(points)
            if fit is None:
                return None
            coefficients, kept, excluded, mape, maximum = fit
            return ShomateCpKernel(
                Tmin=min(T for T, _ in kept),
                Tmax=max(T for T, _ in kept),
                quality=max(0.0, 0.92 - fit_quality_penalty(maximum, mape)),
                source='NIST Chemistry WebBook',
                method='nist_tabulated_shomate_ideal_gas_cp_kernel',
                notes=f'Robust Shomate fit; excluded {len(excluded)}/{len(points)} points',
                fit_mape_percent=mape,
                fit_max_error_percent=maximum,
                source_fingerprint=fingerprint,
                coefficients=tuple(coefficients),
            )


        def _nist_gas_cp_points(
            self,
            tables: Dict[str, Any],
        ) -> list[tuple[float, float]]:
            raw_points = tables.get('gas') or []
            points = []
            for row in raw_points:
                try:
                    if len(row) >= 2:
                        T = float(row[0])
                        Cp = float(row[1])
                        if math.isfinite(T) and T > 0.0 and math.isfinite(Cp) and Cp > 0.0:
                            points.append((T, Cp))
                except (TypeError, ValueError, OverflowError):
                    continue
            return self._clean_cp_points(points)


        @staticmethod
        def _nist_cp_source_fingerprint(tables: Dict[str, Any]) -> str:
            return hashlib.sha256(
                json.dumps(
                    tables,
                    sort_keys=True,
                    separators=(',', ':'),
                    default=str,
                ).encode('utf-8')
            ).hexdigest()


        def _nist_tabulated_shomate_kernel(
            self,
            tables: Dict[str, Any],
            *,
            quality: float,
            method: str,
        ) -> tuple[Optional[ShomateCpKernel], list[tuple[float, float]]]:
            points = self._nist_gas_cp_points(tables)
            fit = self._fit_shomate_cp(points)
            if fit is None:
                return None, []
            coefficients, kept, excluded, mape, maximum = fit
            kernel = ShomateCpKernel(
                Tmin=min(T for T, _ in kept),
                Tmax=max(T for T, _ in kept),
                quality=quality,
                source='NIST Chemistry WebBook',
                method=method,
                notes=(
                    f'Robust Shomate fit to {len(kept)} tabulated Cp points; '
                    f'excluded {len(excluded)}/{len(points)} points'
                ),
                fit_mape_percent=mape,
                fit_max_error_percent=maximum,
                source_fingerprint=self._nist_cp_source_fingerprint(tables),
                coefficients=tuple(coefficients),
            )
            return kernel, kept


        @staticmethod
        def _affine_cp_parameters(
            base_kernel: IdealGasCpKernel,
            points: list[tuple[float, float]],
            *,
            constant_only: bool,
        ) -> Optional[tuple[float, float, float, float]]:
            if not points or any(not base_kernel.covers(T) for T, _ in points):
                return None
            base_values = np.asarray([base_kernel.cp(T) for T, _ in points])
            observed = np.asarray([Cp for _, Cp in points], dtype=float)
            if constant_only:
                intercept = float(np.mean(observed - base_values))
                scale_factor = 1.0
            else:
                design = np.column_stack([np.ones(len(base_values)), base_values])
                if np.linalg.matrix_rank(design) < 2:
                    return None
                intercept, scale_factor = (
                    float(value)
                    for value in np.linalg.lstsq(design, observed, rcond=None)[0]
                )
            predicted = intercept + scale_factor * base_values
            errors = 100.0 * np.abs(predicted / observed - 1.0)
            return (
                intercept,
                scale_factor,
                float(np.mean(errors)),
                float(np.max(errors)),
            )


        def _affine_xtb_cp_kernel(
            self,
            xtb_kernel: IdealGasCpKernel,
            points: list[tuple[float, float]],
            *,
            constant_only: bool,
            quality: float,
            method: str,
            source_fingerprint: str,
        ) -> Optional[AffineIdealGasCpKernel]:
            fit = self._affine_cp_parameters(
                xtb_kernel,
                points,
                constant_only=constant_only,
            )
            if fit is None:
                return None
            intercept, scale_factor, mape, maximum = fit
            try:
                return AffineIdealGasCpKernel(
                    Tmin=xtb_kernel.Tmin,
                    Tmax=xtb_kernel.Tmax,
                    quality=quality,
                    source='NIST Chemistry WebBook + calculated',
                    method=method,
                    notes=(
                        f'{"Constant-residual" if constant_only else "Affine"} '
                        f'calibration of GFN2-xTB RRHO to {len(points)} online '
                        'tabulated Cp points'
                    ),
                    fit_mape_percent=mape,
                    fit_max_error_percent=maximum,
                    source_fingerprint=source_fingerprint,
                    base_kernel=xtb_kernel,
                    intercept=intercept,
                    scale_factor=scale_factor,
                )
            except ValueError:
                return None


        def _piecewise_shomate_affine_xtb_kernel(
            self,
            tables: Dict[str, Any],
            xtb_kernel: IdealGasCpKernel,
        ) -> Optional[PiecewiseIdealGasCpKernel]:
            shomate, kept = self._nist_tabulated_shomate_kernel(
                tables,
                quality=0.94,
                method='nist_tabulated_shomate_segment',
            )
            if shomate is None:
                return None
            fingerprint = hashlib.sha256(
                (
                    self._nist_cp_source_fingerprint(tables)
                    + '|'
                    + xtb_kernel.source_fingerprint
                    + '|piecewise_shomate_affine_xtb_v1'
                ).encode('utf-8')
            ).hexdigest()
            affine = self._affine_xtb_cp_kernel(
                xtb_kernel,
                kept,
                constant_only=False,
                quality=0.94,
                method='nist_affine_xtb_calibration',
                source_fingerprint=fingerprint,
            )
            if affine is None:
                return None

            segments: list[IdealGasCpKernel] = []
            if xtb_kernel.Tmin < shomate.Tmin:
                lower_intercept = (
                    affine.intercept
                    + shomate._cp_native(shomate.Tmin)
                    - affine._cp_native(shomate.Tmin)
                )
                try:
                    segments.append(AffineIdealGasCpKernel(
                        Tmin=xtb_kernel.Tmin,
                        Tmax=shomate.Tmin,
                        quality=0.94,
                        source='NIST Chemistry WebBook + calculated',
                        method='lower_boundary_matched_affine_xtb_cp',
                        notes=f'Affine xTB continuation matched at {shomate.Tmin:g} K',
                        source_fingerprint=fingerprint,
                        base_kernel=xtb_kernel,
                        intercept=lower_intercept,
                        scale_factor=affine.scale_factor,
                    ))
                except ValueError:
                    return None
            segments.append(shomate)
            if xtb_kernel.Tmax > shomate.Tmax:
                upper_intercept = (
                    affine.intercept
                    + shomate._cp_native(shomate.Tmax)
                    - affine._cp_native(shomate.Tmax)
                )
                try:
                    segments.append(AffineIdealGasCpKernel(
                        Tmin=shomate.Tmax,
                        Tmax=xtb_kernel.Tmax,
                        quality=0.94,
                        source='NIST Chemistry WebBook + calculated',
                        method='upper_boundary_matched_affine_xtb_cp',
                        notes=f'Affine xTB continuation matched at {shomate.Tmax:g} K',
                        source_fingerprint=fingerprint,
                        base_kernel=xtb_kernel,
                        intercept=upper_intercept,
                        scale_factor=affine.scale_factor,
                    ))
                except ValueError:
                    return None
            try:
                return PiecewiseIdealGasCpKernel(
                    Tmin=segments[0].Tmin,
                    Tmax=segments[-1].Tmax,
                    quality=0.94,
                    source='NIST Chemistry WebBook + calculated',
                    method='nist_shomate_affine_xtb_piecewise_ideal_gas_cp_kernel',
                    notes=(
                        f'NIST Shomate within {shomate.Tmin:g}-{shomate.Tmax:g} K; '
                        'boundary-matched affine GFN2-xTB RRHO continuation outside'
                    ),
                    fit_mape_percent=shomate.fit_mape_percent,
                    fit_max_error_percent=shomate.fit_max_error_percent,
                    source_fingerprint=fingerprint,
                    segments=tuple(segments),
                    source_Tmin=shomate.Tmin,
                    source_Tmax=shomate.Tmax,
                )
            except ValueError:
                return None


        def _nist_sparse_xtb_kernel(
            self,
            tables: Dict[str, Any],
            xtb_kernel: IdealGasCpKernel,
        ) -> Optional[IdealGasCpKernel]:
            points = self._nist_gas_cp_points(tables)
            fingerprint = hashlib.sha256(
                (
                    self._nist_cp_source_fingerprint(tables)
                    + '|'
                    + xtb_kernel.source_fingerprint
                ).encode('utf-8')
            ).hexdigest()
            if len(points) >= 10:
                return self._piecewise_shomate_affine_xtb_kernel(
                    tables,
                    xtb_kernel,
                )
            if len(points) >= 3:
                return self._affine_xtb_cp_kernel(
                    xtb_kernel,
                    points,
                    constant_only=False,
                    quality=0.93,
                    method='nist_affine_xtb_ideal_gas_cp_kernel',
                    source_fingerprint=fingerprint,
                )
            if points:
                return self._affine_xtb_cp_kernel(
                    xtb_kernel,
                    points,
                    constant_only=True,
                    quality=0.91,
                    method='nist_constant_corrected_xtb_ideal_gas_cp_kernel',
                    source_fingerprint=fingerprint,
                )
            return None


        def _try_xtb_rrho_ideal_gas_cp_kernel(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[IdealGasCpKernel]:
            """Treat every optional xTB provider failure as a normal miss."""
            try:
                return self._xtb_rrho_ideal_gas_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
            except Exception:
                return None


        def resolve_ideal_gas_cp_kernel(
            self,
            symbol: str,
            props: Optional[Dict[str, Any]] = None,
            *,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> Optional[IdealGasCpKernel]:
            """Resolve one executable ideal-gas Cp correlation, not a scalar value."""
            allow_network = self._props_allow_online(props, allow_online)
            props = self._coerce_props(symbol, props, allow_online=allow_network)
            identity = self._ideal_gas_cp_identity(symbol, props)
            fingerprint = self._ideal_gas_cp_props_fingerprint(props)
            xtb_dependencies = (
                tuple(sorted(self._xtb_rrho_dependency_state().items()))
                if allow_estimation
                else ()
            )
            cache_key = (
                identity,
                fingerprint,
                bool(allow_network),
                bool(allow_estimation),
                XTB_RRHO_ARTIFACT_VERSION,
                xtb_dependencies,
            )
            if cache_key in self._ideal_gas_cp_kernel_cache:
                return self._ideal_gas_cp_kernel_cache[cache_key]

            kernel = self._provided_ideal_gas_cp_kernel(props)
            deferred_psi4 = None
            if kernel is None:
                cas = str((props or {}).get('CAS') or (props or {}).get('cas') or '').strip()
                if not cas and re.fullmatch(r'\d{2,7}-\d{2}-\d', str(symbol).strip()):
                    cas = str(symbol).strip()
                if cas:
                    kernel = load_bundled_kernel(cas)
                    if (
                        kernel is not None
                        and kernel.method == 'canonical_psi4_adjusted_ideal_gas_cp'
                    ):
                        deferred_psi4 = replace(
                            kernel,
                            quality=COMPUTATIONAL_RRHO_QUALITY,
                            notes=(
                                f'{kernel.notes}; computational fallback quality '
                                f'{COMPUTATIONAL_RRHO_QUALITY:g}'
                            ),
                        )
                        kernel = None

            if kernel is None:
                kernel = self._get_derived_cp_kernel(
                    NIST_DIRECT_CP_ORIGIN,
                    identity,
                )
            source = None
            if kernel is None:
                source = self._fetch_nist_cp_source(
                    symbol,
                    props,
                    allow_network=allow_network,
                )
                if source:
                    try:
                        kernel = self._kernel_from_nist_source(
                            source,
                            native_only=True,
                        )
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_cp_kernel(
                            NIST_DIRECT_CP_ORIGIN,
                            identity,
                            kernel,
                        )

            points = self._nist_gas_cp_points(source) if source else []
            xtb_kernel = None
            if kernel is None and allow_estimation:
                kernel = self._get_derived_cp_kernel(
                    NIST_XTB_CP_ORIGIN,
                    identity,
                )
                if kernel is None and points:
                    xtb_kernel = self._try_xtb_rrho_ideal_gas_cp_kernel(
                        symbol,
                        props,
                        allow_online=allow_network,
                    )
                if kernel is None and xtb_kernel is not None:
                    try:
                        kernel = self._nist_sparse_xtb_kernel(source, xtb_kernel)
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_cp_kernel(
                            NIST_XTB_CP_ORIGIN,
                            identity,
                            kernel,
                        )

            if kernel is None:
                kernel = self._get_derived_cp_kernel(
                    NIST_SHOMATE_CP_ORIGIN,
                    identity,
                )
                if kernel is None and len(points) >= 10:
                    try:
                        kernel, _kept = self._nist_tabulated_shomate_kernel(
                            source,
                            quality=0.92,
                            method='nist_in_range_shomate_ideal_gas_cp_kernel',
                        )
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_cp_kernel(
                            NIST_SHOMATE_CP_ORIGIN,
                            identity,
                            kernel,
                        )

            if kernel is None and deferred_psi4 is not None:
                kernel = deferred_psi4

            if kernel is None and allow_estimation:
                kernel = xtb_kernel or self._try_xtb_rrho_ideal_gas_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_network,
                )

            if kernel is None:
                kernel = self._get_derived_cp_kernel(
                    NIST_LEGACY_CP_ORIGIN,
                    identity,
                )
                if kernel is None and points:
                    try:
                        kernel = self._kernel_from_nist_source(source)
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_cp_kernel(
                            NIST_LEGACY_CP_ORIGIN,
                            identity,
                            kernel,
                        )

            if kernel is None and allow_estimation:
                kernel = self._atom_increment_ideal_gas_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_network,
                )

            self._ideal_gas_cp_kernel_cache[cache_key] = kernel
            return kernel


        def _kernel_from_nist_liquid_source(
            self,
            tables: Dict[str, Any],
        ) -> Optional[LiquidCpKernel]:
            """Normalize cached/online NIST liquid Cp data to one kernel."""
            fingerprint = hashlib.sha256(
                json.dumps(
                    tables,
                    sort_keys=True,
                    separators=(',', ':'),
                    default=str,
                ).encode('utf-8')
            ).hexdigest()
            segments = tables.get('liquid_shomate') or []
            normalized = []
            for segment in segments:
                try:
                    normalized.append(ShomateLiquidCpKernel(
                        Tmin=float(segment['Tmin_K']),
                        Tmax=float(segment['Tmax_K']),
                        quality=0.975,
                        source='NIST Chemistry WebBook',
                        method='nist_native_shomate_liquid_cp_kernel',
                        notes='Native online NIST liquid Shomate equation',
                        source_fingerprint=fingerprint,
                        coefficients=tuple(float(segment[name]) for name in 'ABCDE'),
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(normalized) == 1:
                return normalized[0]
            if normalized:
                Tmin = min(kernel.Tmin for kernel in normalized)
                Tmax = max(kernel.Tmax for kernel in normalized)

                def evaluate(temperatures):
                    import numpy as np

                    values_T = np.asarray(temperatures, dtype=float)
                    result = np.full(values_T.shape, np.nan, dtype=float)
                    for segment_kernel in normalized:
                        mask = (
                            np.isnan(result)
                            & (values_T >= segment_kernel.Tmin - 1.0e-9)
                            & (values_T <= segment_kernel.Tmax + 1.0e-9)
                        )
                        if np.any(mask):
                            result[mask] = [
                                segment_kernel._cp_native(float(T)) for T in values_T[mask]
                            ]
                    if np.any(np.isnan(result)):
                        valid = np.flatnonzero(~np.isnan(result))
                        if len(valid) < 2:
                            return result
                        result = np.interp(values_T, values_T[valid], result[valid])
                    return result

                return fit_linear_chebyshev_liquid_kernel(
                    evaluate,
                    Tmin,
                    Tmax,
                    quality=0.975,
                    source='NIST Chemistry WebBook',
                    method='nist_multirange_shomate_chebyshev_liquid_cp_kernel',
                    notes=f'Collapsed {len(normalized)} native NIST liquid Shomate ranges',
                    source_fingerprint=fingerprint,
                )

            raw_points = tables.get('liquid') or []
            points = self._clean_cp_points([
                (float(row[0]), float(row[1]))
                for row in raw_points
                if len(row) >= 2
            ])
            if not points:
                return None
            Tmin = points[0][0]
            Tmax = points[-1][0]
            if len(points) == 1:
                point_T, point_cp = points[0]
                return ConstantLiquidCpKernel(
                    Tmin=max(1.0e-9, point_T - STP_POINT_HALF_WIDTH_K),
                    Tmax=point_T + STP_POINT_HALF_WIDTH_K,
                    quality=0.84,
                    source='NIST Chemistry WebBook',
                    method='nist_constant_liquid_cp_kernel',
                    notes=(
                        f'Single tabulated liquid Cp point at {point_T:g} K; native '
                        f'range {point_T - STP_POINT_HALF_WIDTH_K:g}-'
                        f'{point_T + STP_POINT_HALF_WIDTH_K:g} K'
                    ),
                    source_fingerprint=fingerprint,
                    value=point_cp,
                )
            if len(points) < 10:
                linear = self._linear_cp_fit(points)
                if linear is None:
                    return None
                intercept, slope, mape, maximum = linear
                sparse_penalty = (
                    0.01 * max(0.0, maximum - 2.0)
                    + 0.06 * max(0.0, mape - 0.5)
                )
                return PolynomialLiquidCpKernel(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    quality=max(0.0, 0.86 - sparse_penalty),
                    source='NIST Chemistry WebBook',
                    method='nist_linear_liquid_cp_kernel',
                    notes=f'Linear fit to {len(points)} tabulated liquid Cp points',
                    fit_mape_percent=mape,
                    fit_max_error_percent=maximum,
                    source_fingerprint=fingerprint,
                    coefficients=(intercept, slope),
                )
            fit = self._fit_shomate_cp(points)
            if fit is not None:
                coefficients, kept, excluded, mape, maximum = fit
                return ShomateLiquidCpKernel(
                    Tmin=min(T for T, _ in kept),
                    Tmax=max(T for T, _ in kept),
                    quality=max(0.0, 0.92 - fit_quality_penalty(maximum, mape)),
                    source='NIST Chemistry WebBook',
                    method='nist_tabulated_shomate_liquid_cp_kernel',
                    notes=(
                        f'Robust Shomate fit to liquid Cp; excluded '
                        f'{len(excluded)}/{len(points)} points'
                    ),
                    fit_mape_percent=mape,
                    fit_max_error_percent=maximum,
                    source_fingerprint=fingerprint,
                    coefficients=tuple(coefficients),
                )

            def interpolate(temperatures):
                import numpy as np

                values_T = np.asarray(temperatures, dtype=float)
                return np.interp(
                    values_T,
                    np.asarray([T for T, _ in points]),
                    np.asarray([Cp for _, Cp in points]),
                )

            return fit_linear_chebyshev_liquid_kernel(
                interpolate,
                Tmin,
                Tmax,
                quality=0.88,
                source='NIST Chemistry WebBook',
                method='nist_tabulated_chebyshev_liquid_cp_kernel',
                notes=f'Canonical fit to {len(points)} tabulated liquid Cp points',
                source_fingerprint=fingerprint,
            )


        def _kernel_from_nist_solid_source(
            self,
            tables: Dict[str, Any],
            *,
            material_form: str = 'unspecified',
            polymorph: str = '',
        ) -> Optional[SolidCpKernel]:
            """Normalize native online solid tables without smoothing transitions."""
            fingerprint = hashlib.sha256(
                json.dumps(tables, sort_keys=True, separators=(',', ':'), default=str).encode()
            ).hexdigest()
            common = dict(
                quality=0.975, source='NIST Chemistry WebBook',
                notes='Native online NIST solid heat capacity',
                source_fingerprint=fingerprint, material_form=material_form,
                polymorph=polymorph, source_priority=2,
            )
            children = []
            transitions = []
            for segment in tables.get('solid_shomate') or []:
                try:
                    child = ShomateSolidCpKernel(
                        Tmin=float(segment['Tmin_K']), Tmax=float(segment['Tmax_K']),
                        method='nist_native_shomate_solid_cp_kernel',
                        coefficients=tuple(float(segment[name]) for name in 'ABCDE'),
                        **common,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if children:
                    boundary = child.Tmin
                    previous = children[-1]
                    if abs(previous.Tmax - boundary) <= 1.0e-6:
                        first_cp = previous._cp_native(previous.Tmax)
                        second_cp = child._cp_native(child.Tmin)
                        jump = abs(first_cp - second_cp) / max(
                            (abs(first_cp) + abs(second_cp)) / 2.0, 1.0e-30,
                        )
                        if jump > 0.01:
                            transitions.append(boundary)
                children.append(child)
            if len(children) == 1:
                return children[0]
            if children:
                return PiecewiseSolidCpKernel(
                    Tmin=children[0].Tmin, Tmax=children[-1].Tmax,
                    method='nist_piecewise_shomate_solid_cp_kernel',
                    segments=tuple(children), transitions=tuple(transitions),
                    **common,
                )

            raw = sorted(
                (float(row[0]), float(row[1]))
                for row in (tables.get('solid') or [])
                if len(row) >= 2 and float(row[0]) > 0.0 and float(row[1]) > 0.0
            )
            if not raw:
                return None
            if len(raw) == 1:
                point_T, point_cp = raw[0]
                return ConstantSolidCpKernel(
                    Tmin=max(1.0e-9, point_T - STP_POINT_HALF_WIDTH_K),
                    Tmax=point_T + STP_POINT_HALF_WIDTH_K,
                    quality=0.88, source='NIST Chemistry WebBook',
                    method='nist_constant_solid_cp_kernel',
                    notes=f'Single online solid Cp point at {point_T:g} K',
                    source_fingerprint=fingerprint, material_form=material_form,
                    polymorph=polymorph, source_priority=2, value=point_cp,
                )
            table_segments = []
            transition_temperatures = []
            Ts, Cps = [], []
            for T, Cp in raw:
                if Ts and abs(T - Ts[-1]) <= 1.0e-12:
                    if len(Ts) >= 2:
                        table_segments.append((Ts, Cps))
                    transition_temperatures.append(T)
                    Ts, Cps = [T], [Cp]
                else:
                    Ts.append(T)
                    Cps.append(Cp)
            if len(Ts) >= 2:
                table_segments.append((Ts, Cps))
            normalized = [
                TabularSolidCpKernel(
                    Tmin=Ts[0], Tmax=Ts[-1], quality=0.94,
                    source='NIST Chemistry WebBook',
                    method='nist_tabular_solid_cp_kernel',
                    notes=f'Exact linear interpolation of {len(Ts)} online solid Cp points',
                    source_fingerprint=fingerprint, material_form=material_form,
                    polymorph=polymorph, source_priority=2,
                    temperatures=tuple(Ts), values=tuple(Cps),
                )
                for Ts, Cps in table_segments
            ]
            if len(normalized) == 1:
                return normalized[0]
            if normalized:
                return PiecewiseSolidCpKernel(
                    Tmin=normalized[0].Tmin, Tmax=normalized[-1].Tmax,
                    quality=0.94, source='NIST Chemistry WebBook',
                    method='nist_piecewise_tabular_solid_cp_kernel',
                    notes=f'{len(normalized)} online solid phase segments',
                    source_fingerprint=fingerprint, material_form=material_form,
                    polymorph=polymorph, source_priority=2,
                    segments=tuple(normalized), transitions=tuple(transition_temperatures),
                )
            return None


        def _liquid_cp_structure_result(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[PropertyResolutionResult]:
            for key in (
                'smiles', 'SMILES', 'canonical_smiles', 'CanonicalSMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
                'isomeric_smiles', 'IsomericSMILES',
            ):
                result = self._source_result_for_value(
                    props,
                    key,
                    default_source='provided',
                    default_method='provided_smiles',
                    default_quality=1.0,
                )
                if result and result.value:
                    return result
            return self._resolve_smiles_result(
                symbol,
                props,
                allow_online=allow_online,
            )


        @staticmethod
        def _liquid_cp_estimator_quality(
            method_factor: float,
            ideal_gas: IdealGasCpKernel,
            *inputs: PropertyResolutionResult,
        ) -> float:
            qualities = [float(ideal_gas.quality)]
            for result in inputs:
                if result is None or result.value is None:
                    return 0.0
                qualities.append(float(result.quality))
            return max(0.0, min(1.0, float(method_factor) * min(qualities)))


        @staticmethod
        def _liquid_cp_estimator_range(
            ideal_gas: IdealGasCpKernel,
            critical_temperature: float,
        ) -> Optional[tuple[float, float]]:
            Tmin = max(
                float(ideal_gas.Tmin),
                ESTIMATOR_MINIMUM_REDUCED_TEMPERATURE * critical_temperature,
            )
            Tmax = min(
                float(ideal_gas.Tmax),
                ESTIMATOR_MAXIMUM_REDUCED_TEMPERATURE * critical_temperature,
            )
            if not math.isfinite(Tmin) or not math.isfinite(Tmax) or Tmax <= Tmin:
                return None
            return Tmin, Tmax


        def _predictive_liquid_cp_kernel(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
        ) -> Optional[LiquidCpKernel]:
            """Build the benchmarked Bondi/HBD-GC ordinary-liquid estimator."""
            formula = str((props or {}).get('formula') or '').strip()
            if not formula and hasattr(self, '_critical_identity_formula'):
                formula = str(self._critical_identity_formula(props) or '').strip()
            formula_counts = parse_formula_counts(formula) if formula else None
            if formula_counts:
                formula_classification = classify_strict_molecular_organic(
                    cas=(props or {}).get('CAS') or (props or {}).get('cas'),
                    formula=formula,
                )
                if not formula_classification.is_organic:
                    return None
            structure_result = self._liquid_cp_structure_result(
                symbol,
                props,
                allow_online=allow_online,
            )
            smiles = (
                str(structure_result.value).strip()
                if structure_result and structure_result.value else ''
            )
            classification = classify_strict_molecular_organic(
                cas=(props or {}).get('CAS') or (props or {}).get('cas'),
                formula=formula,
                smiles=smiles,
            )
            if not classification.is_organic:
                return None
            formula_result = self._source_result_for_value(
                props,
                'formula',
                value=formula or None,
                default_source='provided',
                default_method='provided_formula',
                default_quality=1.0,
            )
            identity_result = structure_result or formula_result
            if identity_result is None:
                return None
            donor_profile = hydrogen_bond_donor_profile(smiles) if smiles else None
            donor_uncertain = donor_profile is None
            donor_capable_heteroatoms = bool(
                formula_counts
                and any(
                    int(formula_counts.get(element, 0)) > 0
                    for element in ('O', 'N', 'S', 'P', 'Se')
                )
            )
            if donor_profile is None:
                donor_profile = HydrogenBondDonorProfile()

            ideal_gas = self.resolve_ideal_gas_cp_kernel(
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=True,
            )
            if ideal_gas is None:
                return None
            try:
                critical = self.resolve_critical_properties(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=True,
                )
            except Exception:
                return None
            tc_result = (critical or {}).get('Tc')
            if tc_result is None or tc_result.value is None:
                return None
            try:
                Tc = float(tc_result.value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(Tc) or Tc <= 0.0:
                return None
            if float(tc_result.quality) < MINIMUM_ESTIMATOR_CRITICAL_QUALITY:
                return None
            temperature_range = self._liquid_cp_estimator_range(ideal_gas, Tc)
            if temperature_range is None:
                return None
            Tmin, Tmax = temperature_range

            mixed_donors = (
                len(donor_profile.all_classes) > 1
                or bool(donor_profile.other)
                or (donor_uncertain and donor_capable_heteroatoms)
            )
            gc_fractions = donor_profile.gc_fractions()
            use_gc = (
                donor_profile.onh_count > 0
                and not mixed_donors
                and gc_fractions is not None
            )

            common_fingerprint = {
                'ideal_gas': ideal_gas.source_fingerprint,
                'Tc': Tc,
                'formula': formula,
                'smiles': smiles,
                'donors': donor_profile.__dict__,
            }
            if use_gc:
                method_factor = HBD_RATIO_GC_QUALITY_FACTOR
                quality = self._liquid_cp_estimator_quality(
                    method_factor,
                    ideal_gas,
                    tc_result,
                    identity_result,
                )
                onh_count = donor_profile.onh_count
                fractions = gc_fractions

                def evaluate(temperatures):
                    import numpy as np

                    values_T = np.asarray(temperatures, dtype=float)
                    flat = values_T.reshape(-1)
                    cp_ideal = np.asarray(
                        [ideal_gas.cp(float(T)) for T in flat], dtype=float,
                    ).reshape(values_T.shape)
                    Tr = values_T / Tc
                    size = np.log(cp_ideal / (R * onh_count))
                    y = (
                        -0.696566247027313
                        + 0.132964405534084 * size
                        - 0.422120326414122 * Tr
                        + 0.227599743706810 * math.log(onh_count)
                        + fractions['polyol'] * (
                            -0.318135013747906
                            + 0.0416525663126231 * size
                            + 0.263525226317632 * Tr
                        )
                        + fractions['nitrogen'] * (
                            -0.709441931360163
                            + 0.172563233629678 * size
                            + 0.613536310423229 * Tr
                        )
                        + fractions['acid'] * (
                            -0.181137278384696
                            + 0.0508194424909603 * size
                            + 0.209074346990637 * Tr
                        )
                        + fractions['phenol'] * (
                            -0.942545987840754
                            + 0.260531391769113 * size
                            + 0.455449212252080 * Tr
                        )
                    )
                    ratio = np.clip(
                        np.exp(y),
                        HBD_RATIO_MINIMUM,
                        HBD_RATIO_MAXIMUM,
                    )
                    return cp_ideal / ratio

                donor_note = (
                    f'O/N donor classes={donor_profile.onh_classes}; '
                    f'N_ONH={onh_count}; fractions={fractions}'
                )
                return fit_linear_chebyshev_liquid_kernel(
                    evaluate,
                    Tmin,
                    Tmax,
                    quality=quality,
                    source='estimated',
                    method='hbd_ratio_gc_liquid_cp_kernel',
                    notes=(
                        'HBD organic direct Cpig/Cpl log-ratio group contribution; '
                        'held-out O/N-organic mean MARD 6.03%, case median 4.73%; '
                        f'{donor_note}; ideal-gas source={ideal_gas.method}; '
                        f'Tc source={tc_result.method}; identity source={identity_result.method}'
                    ),
                    source_fingerprint=hashlib.sha256(
                        json.dumps(
                            {**common_fingerprint, 'model': 'hbd_ratio_gc_v1'},
                            sort_keys=True,
                            default=str,
                        ).encode('utf-8')
                    ).hexdigest(),
                )

            omega_result = (critical or {}).get('omega')
            if omega_result is None or omega_result.value is None:
                return None
            try:
                omega = float(omega_result.value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(omega) or not -0.5 <= omega <= 2.0:
                return None
            if float(omega_result.quality) < MINIMUM_ESTIMATOR_CRITICAL_QUALITY:
                return None
            method_factor = (
                MIXED_DONOR_BONDI_QUALITY_FACTOR
                if mixed_donors
                else ROWLINSON_BONDI_QUALITY_FACTOR
            )
            quality = self._liquid_cp_estimator_quality(
                method_factor,
                ideal_gas,
                tc_result,
                omega_result,
                identity_result,
            )

            def evaluate(temperatures):
                import numpy as np

                values_T = np.asarray(temperatures, dtype=float)
                flat = values_T.reshape(-1)
                cp_ideal = np.asarray(
                    [ideal_gas.cp(float(T)) for T in flat], dtype=float,
                ).reshape(values_T.shape)
                Tr = values_T / Tc
                one_minus_Tr = 1.0 - Tr
                departure = R * (
                    1.45
                    + 0.45 / one_minus_Tr
                    + omega * (
                        4.2775
                        + 6.3 * np.cbrt(one_minus_Tr) / Tr
                        + 0.4355 / one_minus_Tr
                    )
                )
                return cp_ideal + departure

            method = (
                'mixed_donor_rowlinson_bondi_liquid_cp_kernel'
                if mixed_donors
                else 'rowlinson_bondi_liquid_cp_kernel'
            )
            return fit_linear_chebyshev_liquid_kernel(
                evaluate,
                Tmin,
                Tmax,
                quality=quality,
                source='estimated',
                method=method,
                notes=(
                    'Rowlinson-Bondi ordinary-liquid Cp estimate; '
                    f'{"provisional mixed-donor" if mixed_donors else "non-HBD or thiol-only"} '
                    'quality tier; non-HBD-organic held-out mean MARD 3.57%, '
                    'case median 1.97%; '
                    f'donor classes={donor_profile.all_classes}; '
                    f'ideal-gas source={ideal_gas.method}; Tc source={tc_result.method}; '
                    f'omega source={omega_result.method}; identity source={identity_result.method}; '
                    f'donor classification={"unavailable" if donor_uncertain else "RDKit"}'
                ),
                source_fingerprint=hashlib.sha256(
                    json.dumps(
                        {
                            **common_fingerprint,
                            'model': 'rowlinson_bondi_v1',
                            'omega': omega,
                            'mixed_donors': mixed_donors,
                        },
                        sort_keys=True,
                        default=str,
                    ).encode('utf-8')
                ).hexdigest(),
            )


        def resolve_liquid_cp_kernel(
            self,
            symbol: str,
            props: Optional[Dict[str, Any]] = None,
            *,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> Optional[LiquidCpKernel]:
            """Resolve one reusable ordinary-liquid heat-capacity correlation."""
            allow_network = self._props_allow_online(props, allow_online)
            props = self._coerce_props(symbol, props, allow_online=allow_network)
            identity = self._ideal_gas_cp_identity(symbol, props)
            provided_range = self._provided_liquid_cp_range(
                symbol,
                props,
                allow_online=allow_network,
                allow_estimation=allow_estimation,
            )
            fingerprint = self._liquid_cp_props_fingerprint(props, provided_range)
            cache_key = (identity, fingerprint, bool(allow_network), bool(allow_estimation))
            if cache_key in self._liquid_cp_kernel_cache:
                return self._liquid_cp_kernel_cache[cache_key]

            kernel = self._provided_liquid_cp_kernel(props, provided_range)
            if kernel is None:
                cas = str((props or {}).get('CAS') or (props or {}).get('cas') or '').strip()
                if not cas and re.fullmatch(r'\d{2,7}-\d{2}-\d', str(symbol).strip()):
                    cas = str(symbol).strip()
                if not cas:
                    # Exact bundled name/formula lookup avoids cold-loading
                    # Perry merely to recover a CAS. Formula matches are used
                    # only when unique in the liquid database.
                    for candidate in (
                        symbol,
                        (props or {}).get('symbol'),
                        (props or {}).get('name'),
                        (props or {}).get('formula'),
                    ):
                        cas = lookup_bundled_liquid_cas(candidate)
                        if cas:
                            break
                if cas:
                    kernel = load_bundled_liquid_kernel(cas)

            # A chemicals.json Cp_liquid scalar is only one STP point and must
            # not shadow the broader, audited canonical database.
            if kernel is None:
                kernel = self._stored_liquid_cp_point_kernel(props)

            if kernel is None:
                kernel = self._get_derived_liquid_cp_kernel('online', identity)
            if kernel is None:
                source = self._fetch_nist_cp_source(
                    symbol,
                    props,
                    allow_network=allow_network,
                )
                if source:
                    try:
                        kernel = self._kernel_from_nist_liquid_source(source)
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_liquid_cp_kernel('online', identity, kernel)

            if kernel is None and allow_estimation:
                kernel = self._predictive_liquid_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_network,
                )

            if kernel is None and allow_estimation:
                ideal_gas = self.resolve_ideal_gas_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_network,
                    allow_estimation=True,
                )
                if ideal_gas is not None:
                    kernel = ScaledIdealGasLiquidCpKernel(
                        Tmin=ideal_gas.Tmin,
                        Tmax=ideal_gas.Tmax,
                        quality=SCALED_IDEAL_GAS_QUALITY_FACTOR * ideal_gas.quality,
                        source='estimated',
                        method='scaled_ideal_gas_liquid_cp_kernel',
                        notes=(
                            'Fallback to 1.3x resolved ideal-gas Cp with '
                            f'quality multiplier {SCALED_IDEAL_GAS_QUALITY_FACTOR:g}; '
                            f'ideal-gas source: {ideal_gas.method}'
                        ),
                        source_fingerprint=hashlib.sha256(
                            f'1.3|{ideal_gas.source_fingerprint}'.encode('utf-8')
                        ).hexdigest(),
                        ideal_gas_kernel=ideal_gas,
                        scale_factor=1.3,
                    )

            self._liquid_cp_kernel_cache[cache_key] = kernel
            return kernel


        def _modified_kopp_solid_cp_kernel(
            self, symbol: str, props: Dict[str, Any],
        ) -> Optional[ModifiedKoppSolidCpKernel]:
            formula = (props or {}).get('formula') or (props or {}).get('Formula')
            if formula and re.search(r'(?:[+-]\d*|\(\d*[+-]\))$', str(formula).strip()):
                return None
            counts = self._ideal_gas_cp_formula_counts(formula)
            if not counts:
                return None
            value = sum(
                count * MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K.get(
                    element, MODIFIED_KOPP_OTHER_J_MOL_K,
                )
                for element, count in counts.items()
            )
            if not math.isfinite(value) or value <= 0.0:
                return None
            organic = classify_strict_molecular_organic(
                cas=(props or {}).get('CAS') or (props or {}).get('cas'),
                formula=formula,
                smiles=(props or {}).get('smiles'),
            ).is_organic
            quality = 0.72 if organic else 0.64
            benchmark = (
                '102 strict organic CRC cases: median 6.75%, mean 9.27%, P90 22.14%'
                if organic else
                '127 Perry-formula/CRC cases: median 6.10%, mean 8.37%, P90 18.27%'
            )
            form, polymorph = self._solid_form_selection(props)
            fingerprint = hashlib.sha256(
                json.dumps({'model': 'modified_kopp_1992', 'counts': counts}, sort_keys=True).encode()
            ).hexdigest()
            return ModifiedKoppSolidCpKernel(
                Tmin=REFERENCE_TEMPERATURE_K - STP_POINT_HALF_WIDTH_K,
                Tmax=REFERENCE_TEMPERATURE_K + STP_POINT_HALF_WIDTH_K,
                quality=quality, source='estimated',
                method='modified_kopp_solid_cp_kernel',
                notes=(
                    'Hurst-Harrison modified Kopp estimate, valid only near 298.15 K; '
                    f'{benchmark}; formula={formula}'
                ),
                source_fingerprint=fingerprint, material_form=form,
                polymorph=polymorph, source_priority=50, value=value,
                atom_counts=tuple(sorted((element, float(count)) for element, count in counts.items())),
            )


        def _lastovka_solid_cp_kernel(
            self, symbol: str, props: Dict[str, Any],
        ) -> Optional[LastovkaSolidCpKernel]:
            formula = (props or {}).get('formula') or (props or {}).get('Formula')
            counts = self._ideal_gas_cp_formula_counts(formula)
            if not counts or not classify_strict_molecular_organic(
                cas=(props or {}).get('CAS') or (props or {}).get('cas'),
                formula=formula,
                smiles=(props or {}).get('smiles'),
            ).is_organic:
                return None
            allowed = {'C', 'H', 'N', 'O', 'S'}
            if not set(counts) <= allowed:
                return None
            try:
                from chemicals.elements import periodic_table
                molecular_weight = float((props or {}).get('MW'))
                formula_weight = sum(periodic_table[element].MW * count for element, count in counts.items())
            except (TypeError, ValueError, KeyError):
                return None
            if not (12.24 <= molecular_weight <= 402.4):
                return None
            if abs(formula_weight / molecular_weight - 1.0) > 0.02:
                return None
            limits = {
                'C': (0.613, 0.952), 'H': (0.0373, 0.152),
                'N': (0.0, 0.154), 'O': (0.0, 0.188), 'S': (0.0, 0.296),
            }
            fractions = {
                element: periodic_table[element].MW * counts.get(element, 0) / molecular_weight
                for element in allowed
            }
            if any(not (low <= fractions[element] <= high) for element, (low, high) in limits.items()):
                return None
            transition = None
            for key in ('Tt', 'Tm'):
                try:
                    value = float((props or {}).get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 100.0:
                    transition = value
                    break
            if transition is None:
                return None
            alpha = sum(counts.values()) / molecular_weight
            form, polymorph = self._solid_form_selection(props)
            return LastovkaSolidCpKernel(
                Tmin=100.0, Tmax=transition,
                quality=0.70, source='estimated',
                method='lastovka_solid_cp_kernel',
                notes=(
                    'Lastovka-Fulem-Becerra-Shaw temperature-dependent solid-organic '
                    'estimate within its published MW and elemental mass-fraction domain; '
                    '38 identity-confirmed CRC points at 298.15 K: median 6.38%, '
                    'mean 6.63%, P90 13.11%; upper range limited to supplied Tt/Tm'
                ),
                source_fingerprint=hashlib.sha256(
                    json.dumps({'model': 'lastovka_solid_2008', 'counts': counts, 'MW': molecular_weight, 'Tmax': transition}, sort_keys=True).encode()
                ).hexdigest(),
                material_form=form, polymorph=polymorph, source_priority=51,
                similarity_variable=alpha, molecular_weight=molecular_weight,
            )


        def resolve_solid_cp_kernel(
            self,
            symbol: str,
            props: Optional[Dict[str, Any]] = None,
            *,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> Optional[SolidCpKernel]:
            """Resolve one reusable, material-form-aware solid Cp kernel."""
            allow_network = self._props_allow_online(props, allow_online)
            props = self._coerce_props(symbol, props, allow_online=allow_network)
            identity = self._ideal_gas_cp_identity(symbol, props)
            provided_range = self._provided_solid_cp_range(
                symbol,
                props,
                allow_online=allow_network,
            )
            fingerprint = self._solid_cp_props_fingerprint(props, provided_range)
            cache_key = (identity, fingerprint, bool(allow_network), bool(allow_estimation))
            if cache_key in self._solid_cp_kernel_cache:
                return self._solid_cp_kernel_cache[cache_key]

            form, polymorph = self._solid_form_selection(props)
            derived_identity = f'{identity}|{form}|{polymorph.casefold()}'
            kernel = self._provided_solid_cp_kernel(props, provided_range)
            if kernel is None:
                cas = str((props or {}).get('CAS') or (props or {}).get('cas') or '').strip()
                if not cas and re.fullmatch(r'\d{2,7}-\d{2}-\d', str(symbol).strip()):
                    cas = str(symbol).strip()
                if not cas:
                    for candidate in (
                        symbol, (props or {}).get('symbol'),
                        (props or {}).get('name'), (props or {}).get('formula'),
                    ):
                        cas = lookup_bundled_solid_cas(candidate)
                        if cas:
                            break
                if cas:
                    kernel = load_bundled_solid_kernel(
                        cas, material_form=form, polymorph=polymorph,
                    )

            if kernel is None:
                kernel = self._stored_solid_cp_point_kernel(props)
            if kernel is None:
                kernel = self._get_derived_solid_cp_kernel('online', derived_identity)
            if kernel is None:
                source = self._fetch_nist_cp_source(
                    symbol, props, allow_network=allow_network,
                )
                if source:
                    try:
                        kernel = self._kernel_from_nist_solid_source(
                            source, material_form=form, polymorph=polymorph,
                        )
                    except (TypeError, ValueError, OverflowError):
                        kernel = None
                    if kernel is not None:
                        self._set_derived_solid_cp_kernel('online', derived_identity, kernel)

            if kernel is None and allow_estimation:
                kopp = self._modified_kopp_solid_cp_kernel(symbol, props)
                lastovka = self._lastovka_solid_cp_kernel(symbol, props)
                kernel = lastovka or kopp

            self._solid_cp_kernel_cache[cache_key] = kernel
            return kernel


        def resolve_heat_capacity(
            self,
            symbol: str,
            T: float,
            phase: str = 'liquid',
            props: Dict[str, Any] = None,
            allow_online: bool = True,
        ) -> PropertyResolutionResult:
            """
            Resolve heat capacity at given temperature.

            Args:
                symbol: Chemical symbol
                T: Temperature [K]
                phase: 'liquid', 'vapor', or 'ideal_gas'
                props: Dict with Cp_coeffs if available

            Returns:
                PropertyResolutionResult with Cp in J/mol-K
            """
            allow_online = self._props_allow_online(props, allow_online)
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            phase_key = phase.strip().lower().replace('-', '_')

            if self._ideal_gas_phase(phase_key):
                kernel = self.resolve_ideal_gas_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if kernel is not None:
                    evaluation = kernel.evaluate(T)
                    notes = kernel.notes
                    if evaluation.range_note:
                        notes = f'{notes}; {evaluation.range_note}' if notes else evaluation.range_note
                    if kernel.method == 'provided_cubic_ideal_gas_cp_kernel':
                        scalar_method = 'polynomial'
                    elif kernel.method.startswith('provided_'):
                        scalar_method = 'provided_heat_capacity_fit'
                    else:
                        scalar_method = kernel.method
                    return PropertyResolutionResult(
                        value=evaluation.value,
                        source=kernel.source,
                        method=scalar_method,
                        quality=evaluation.quality,
                        notes=notes,
                    )
                raise PropertyResolutionError(
                    f"Cannot determine ideal-gas heat capacity for '{symbol}' at "
                    f"T={T:.1f} K from provided, bundled, cached online, online, "
                    "or estimated correlations."
                )

            if phase_key == 'liquid':
                kernel = self.resolve_liquid_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if kernel is not None:
                    evaluation = kernel.evaluate(T)
                    notes = kernel.notes
                    if evaluation.range_note:
                        notes = f'{notes}; {evaluation.range_note}' if notes else evaluation.range_note
                    if kernel.method.startswith('provided_'):
                        scalar_method = 'provided_heat_capacity_fit'
                    else:
                        scalar_method = kernel.method
                    return PropertyResolutionResult(
                        value=evaluation.value,
                        source=kernel.source,
                        method=scalar_method,
                        quality=evaluation.quality,
                        notes=notes,
                    )
                raise PropertyResolutionError(
                    f"Cannot determine ordinary-liquid heat capacity for '{symbol}' at "
                    f"T={T:.1f} K from provided, bundled, cached online, online, "
                    "or estimated correlations."
                )

            if phase_key in {'solid', 'crystal', 'crystalline'}:
                kernel = self.resolve_solid_cp_kernel(
                    symbol,
                    props,
                    allow_online=allow_online,
                )
                if kernel is not None:
                    evaluation = kernel.evaluate(T)
                    active = kernel.active_kernel(T)
                    notes = active.notes
                    if evaluation.range_note:
                        notes = f'{notes}; {evaluation.range_note}' if notes else evaluation.range_note
                    return PropertyResolutionResult(
                        value=evaluation.value,
                        source=active.source,
                        method=(
                            'provided_heat_capacity_fit'
                            if active.method.startswith('provided_')
                            else active.method
                        ),
                        quality=evaluation.quality,
                        notes=notes,
                    )
                raise PropertyResolutionError(
                    f"Cannot determine solid heat capacity for '{symbol}' at T={T:.1f} K "
                    "from provided, bundled, cached online, online, or admitted "
                    "solid estimators."
                )

            raise PropertyResolutionError(
                f"Unsupported heat-capacity phase {phase!r} for '{symbol}'."
            )


        def _fetch_cp_online(
            self,
            symbol: str,
            T: float,
            phase: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[HeatCapacityLookup]:
            """Fetch tabulated heat capacity from online sources."""
            for candidate in self._identifier_candidates(symbol, props):
                try:
                    lookup = self._fetch_cp_nist(candidate, T, phase)
                except LookupError:
                    continue
                if lookup:
                    return lookup
            return None


        def _fetch_cp_nist(self, identifier: str, T: float, phase: str) -> Optional[HeatCapacityLookup]:
            """Fetch NIST tabulated heat-capacity data and interpolate at T."""
            cache_key = f"cp_nist_v3_{identifier}"
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                return self._evaluate_nist_cp_tables(cached, T, phase)

            transient_failure = False
            for query_field in ('Name', 'Formula'):
                try:
                    params = urllib.parse.urlencode({
                        query_field: identifier,
                        'Units': 'SI',
                        'Mask': '3',
                    })
                    req = urllib.request.Request(f"{self.NIST_WEBBOOK}?{params}")
                    req.add_header('User-Agent', 'PFD-Editor/1.0')
                    with urllib.request.urlopen(req, timeout=15) as response:
                        html = response.read().decode('utf-8', errors='ignore')

                    tables = self._parse_nist_cp_tables(html)
                    if tables:
                        self._set_cache(cache_key, tables)
                        return self._evaluate_nist_cp_tables(tables, T, phase)
                except Exception as e:
                    transient_failure = transient_failure or self._is_transient_lookup_error(e)

            if transient_failure:
                raise LookupError(f"Transient NIST heat-capacity lookup failure for '{identifier}'")
            self._set_missing_cache(cache_key)
            return None


        def integrate_cp_online(
            self,
            symbol: str,
            T1: float,
            T2: float,
            phase: str,
            props: Optional[Dict[str, Any]] = None,
        ) -> Optional[HeatCapacityIntegralLookup]:
            """Return an analytic online Cp integral in kJ/mol when available."""
            if abs(T2 - T1) < 1e-12:
                phase_key = phase.strip().lower().replace('-', '_')
                table_phase = 'gas' if phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'} else phase_key
                return HeatCapacityIntegralLookup(
                    value=0.0,
                    phase=table_phase,
                    T1=T1,
                    T2=T2,
                    method='zero_temperature_span',
                    quality=1.0,
                )
            for candidate in self._identifier_candidates(symbol, props):
                try:
                    integral = self._integrate_cp_nist(candidate, T1, T2, phase)
                except LookupError:
                    continue
                if integral:
                    return integral
            return None


        def _integrate_cp_nist(
            self,
            identifier: str,
            T1: float,
            T2: float,
            phase: str,
        ) -> Optional[HeatCapacityIntegralLookup]:
            cache_key = f"cp_nist_v3_{identifier}"
            cached = self._get_cache(cache_key)
            if cached:
                if self._is_missing_cache(cached):
                    return None
                return self._integrate_nist_cp_tables(cached, T1, T2, phase)

            transient_failure = False
            for query_field in ('Name', 'Formula'):
                try:
                    params = urllib.parse.urlencode({
                        query_field: identifier,
                        'Units': 'SI',
                        'Mask': '3',
                    })
                    req = urllib.request.Request(f"{self.NIST_WEBBOOK}?{params}")
                    req.add_header('User-Agent', 'PFD-Editor/1.0')
                    with urllib.request.urlopen(req, timeout=15) as response:
                        html = response.read().decode('utf-8', errors='ignore')

                    tables = self._parse_nist_cp_tables(html)
                    if tables:
                        self._set_cache(cache_key, tables)
                        return self._integrate_nist_cp_tables(tables, T1, T2, phase)
                except Exception as e:
                    transient_failure = transient_failure or self._is_transient_lookup_error(e)

            if transient_failure:
                raise LookupError(f"Transient NIST heat-capacity integral lookup failure for '{identifier}'")
            self._set_missing_cache(cache_key)
            return None


        def _parse_nist_cp_tables(self, html: str) -> Dict[str, list[list[float]]]:
            """Parse NIST native Shomate equations and tabulated Cp rows."""
            parsed: Dict[str, list[tuple[float, float]]] = {
                'gas': [], 'liquid': [], 'solid': [],
            }
            shomate_segments: Dict[str, list[dict[str, float]]] = {
                'gas': [],
                'liquid': [],
                'solid': [],
            }

            for label, rows in self._html_tables(html):
                label_lower = label.strip().lower()
                if 'shomate' in label_lower:
                    if 'gas' in label_lower:
                        shomate_segments['gas'].extend(self._parse_nist_shomate_rows(rows))
                    elif 'liquid' in label_lower:
                        shomate_segments['liquid'].extend(self._parse_nist_shomate_rows(rows))
                    elif 'solid' in label_lower:
                        shomate_segments['solid'].extend(self._parse_nist_shomate_rows(rows))
                    continue
                if label_lower == 'constant pressure heat capacity of gas':
                    phase = 'gas'
                elif label_lower == 'constant pressure heat capacity of liquid':
                    phase = 'liquid'
                elif label_lower == 'constant pressure heat capacity of solid':
                    phase = 'solid'
                else:
                    continue
                if not rows:
                    continue

                header = ' '.join(rows[0]).lower()
                if 'j/mol' not in header or 'temperature' not in header or '(k)' not in header:
                    continue

                for row in rows[1:]:
                    if len(row) < 2:
                        continue
                    Cp = self._first_number(row[0])
                    row_T = self._first_number(row[1])
                    if Cp is None or row_T is None:
                        continue
                    if 1.0 <= Cp <= 1000.0 and 1.0 <= row_T <= 7000.0:
                        parsed[phase].append((float(row_T), float(Cp)))

            result: Dict[str, list[list[float]]] = {}
            for phase, points in parsed.items():
                if phase == 'solid':
                    # Duplicate temperatures with different Cp values encode
                    # opposite sides of a solid transition and must survive.
                    cleaned = sorted(set(points), key=lambda item: (item[0], item[1]))
                else:
                    cleaned = self._clean_cp_points(points)
                if cleaned:
                    result[phase] = [[T, Cp] for T, Cp in cleaned]
            if shomate_segments['gas']:
                result['gas_shomate'] = shomate_segments['gas']
            if shomate_segments['liquid']:
                result['liquid_shomate'] = shomate_segments['liquid']
            if shomate_segments['solid']:
                result['solid_shomate'] = shomate_segments['solid']
            return result


        def _parse_nist_shomate_rows(
            self,
            rows: list[list[str]],
        ) -> list[dict[str, float]]:
            """Normalize one WebBook Shomate coefficient table."""
            if not rows or len(rows[0]) < 2:
                return []
            ranges = []
            for cell in rows[0][1:]:
                numbers = re.findall(
                    r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?',
                    html_module.unescape(str(cell)).replace('−', '-').replace(',', ''),
                )
                if len(numbers) < 2:
                    ranges.append(None)
                    continue
                try:
                    Tmin, Tmax = float(numbers[0]), float(numbers[1])
                except ValueError:
                    ranges.append(None)
                    continue
                ranges.append((Tmin, Tmax) if 0.0 < Tmin < Tmax else None)

            coefficients: dict[str, list[Optional[float]]] = {}
            for row in rows[1:]:
                if not row:
                    continue
                name = str(row[0]).strip().upper()
                if name not in set('ABCDE'):
                    continue
                values = []
                for cell in row[1:1 + len(ranges)]:
                    values.append(self._first_number(cell))
                coefficients[name] = values

            segments = []
            for index, limits in enumerate(ranges):
                if limits is None:
                    continue
                values = {}
                valid = True
                for name in 'ABCDE':
                    entries = coefficients.get(name) or []
                    value = entries[index] if index < len(entries) else None
                    if value is None or not math.isfinite(float(value)):
                        valid = False
                        break
                    values[name] = float(value)
                if valid:
                    segments.append({
                        'Tmin_K': limits[0],
                        'Tmax_K': limits[1],
                        **values,
                    })
            return segments


        def _clean_cp_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            """Merge duplicate NIST Cp temperatures and remove isolated local outliers."""
            if not points:
                return []

            binned: Dict[float, list[float]] = {}
            for T, Cp in points:
                binned.setdefault(round(float(T), 2), []).append(float(Cp))
            merged = sorted((T, self._median(values)) for T, values in binned.items())
            if len(merged) < 4:
                return merged

            cleaned = []
            for index, (T, Cp) in enumerate(merged):
                neighbors = [
                    other_Cp
                    for other_index, (other_T, other_Cp) in enumerate(merged)
                    if other_index != index and abs(other_T - T) <= 25.0
                ]
                if len(neighbors) < 3:
                    cleaned.append((T, Cp))
                    continue
                local_median = self._median(neighbors)
                deviations = [abs(value - local_median) for value in neighbors]
                local_mad = self._median(deviations) or 1e-12
                if abs(Cp - local_median) <= max(10.0, 4.0 * 1.4826 * local_mad):
                    cleaned.append((T, Cp))
            return cleaned


        def _cp_lookup_result(
            self,
            value: float,
            phase: str,
            T: float,
            method: str,
            quality: float,
            notes: str,
        ) -> Optional[HeatCapacityLookup]:
            if not (1.0 <= value <= 1000.0):
                return None
            warning = 'Estimated from NIST tabulated Cp data; '
            return HeatCapacityLookup(
                value=value,
                phase=phase,
                T=T,
                method=method,
                quality=quality,
                notes=warning + notes,
            )


        @staticmethod
        def _least_squares(coeff_rows: list[list[float]], values: list[float]) -> Optional[list[float]]:
            n_terms = len(coeff_rows[0]) if coeff_rows else 0
            if n_terms == 0 or len(coeff_rows) < n_terms:
                return None
            normal = []
            rhs = []
            for i in range(n_terms):
                normal.append([
                    sum(row[i] * row[j] for row in coeff_rows)
                    for j in range(n_terms)
                ])
                rhs.append(sum(value * row[i] for row, value in zip(coeff_rows, values)))
            matrix = [row[:] + [rhs[index]] for index, row in enumerate(normal)]
            for col in range(n_terms):
                pivot = max(range(col, n_terms), key=lambda row: abs(matrix[row][col]))
                if abs(matrix[pivot][col]) < 1e-12:
                    return None
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
                divisor = matrix[col][col]
                matrix[col] = [value / divisor for value in matrix[col]]
                for row in range(n_terms):
                    if row == col:
                        continue
                    factor = matrix[row][col]
                    matrix[row] = [
                        matrix[row][index] - factor * matrix[col][index]
                        for index in range(n_terms + 1)
                    ]
            return [matrix[index][-1] for index in range(n_terms)]


        @staticmethod
        def _linear_cp_fit(points: list[tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
            x_values = [T for T, _ in points]
            y_values = [Cp for _, Cp in points]
            fit = VaporPressureMixin._linear_fit(x_values, y_values)
            if fit is None:
                return None
            slope, intercept = fit
            errors = [
                abs((intercept + slope * T - Cp) / Cp) * 100.0
                for T, Cp in points
                if Cp
            ]
            return intercept, slope, sum(errors) / len(errors), max(errors)


        @staticmethod
        def _shomate_cp_basis(T: float) -> list[float]:
            t = T / 1000.0
            return [1.0, t, t * t, t * t * t, 1.0 / (t * t)]


        @classmethod
        def _relative_shomate_cp_fit(
            cls,
            points: list[tuple[float, float]],
        ) -> Optional[list[float]]:
            if len(points) < 5:
                return None
            design = np.asarray([
                cls._shomate_cp_basis(T) for T, _ in points
            ], dtype=float)
            values = np.asarray([Cp for _, Cp in points], dtype=float)
            relative_design = design / values[:, None]
            scales = np.sqrt(np.mean(relative_design * relative_design, axis=0))
            if np.any(~np.isfinite(scales)) or np.any(scales <= 1.0e-14):
                return None
            scaled = np.linalg.lstsq(
                relative_design / scales,
                np.ones(len(values)),
                rcond=1.0e-10,
            )[0]
            coefficients = scaled / scales
            if np.any(~np.isfinite(coefficients)):
                return None
            return [float(value) for value in coefficients]


        def _fit_shomate_cp(
            self,
            points: list[tuple[float, float]],
        ) -> Optional[tuple[list[float], list[tuple[float, float]], list[tuple[float, float]], float, float]]:
            if len(points) < 10:
                return None

            kept = [True] * len(points)
            coefficients = None
            for _ in range(8):
                fit_points = [point for point, include in zip(points, kept) if include]
                if len(fit_points) < 10:
                    return None
                coefficients = self._relative_shomate_cp_fit(fit_points)
                if coefficients is None:
                    return None

                residuals = []
                for T, Cp in points:
                    predicted = sum(coef * basis for coef, basis in zip(coefficients, self._shomate_cp_basis(T)))
                    residuals.append(math.log(Cp) - math.log(max(predicted, 1e-9)))

                kept_abs = [abs(residual) for residual, include in zip(residuals, kept) if include]
                median_abs = self._median(kept_abs)
                mad = self._median([abs(value - median_abs) for value in kept_abs]) or 1e-12
                threshold = max(0.06, median_abs + 3.0 * 1.4826 * mad)
                candidates = [index for index, include in enumerate(kept) if include and abs(residuals[index]) > threshold]
                if not candidates:
                    break

                new_kept = kept[:]
                for index in candidates:
                    new_kept[index] = False
                minimum_kept = max(10, math.ceil(0.8 * len(points)))
                if sum(new_kept) < minimum_kept:
                    worst = max(
                        [index for index, include in enumerate(kept) if include],
                        key=lambda index: abs(residuals[index]),
                    )
                    new_kept = kept[:]
                    new_kept[worst] = False
                if new_kept == kept:
                    break
                kept = new_kept

            kept_points = [point for point, include in zip(points, kept) if include]
            coefficients = self._relative_shomate_cp_fit(kept_points)
            if coefficients is None:
                return None

            errors = []
            for T, Cp in kept_points:
                predicted = sum(coef * basis for coef, basis in zip(coefficients, self._shomate_cp_basis(T)))
                if not (1.0 <= predicted <= 1000.0):
                    return None
                errors.append(abs((predicted - Cp) / Cp) * 100.0)
            if not errors:
                return None
            excluded = [point for point, include in zip(points, kept) if not include]
            return coefficients, kept_points, excluded, sum(errors) / len(errors), max(errors)


        @staticmethod
        def _interpolate_or_clamp_cp(points: list[tuple[float, float]], T: float) -> tuple[float, str]:
            if len(points) == 1:
                point_T, point_Cp = points[0]
                return point_Cp, f"single point at {point_T:g} K"
            lower = [point for point in points if point[0] <= T]
            upper = [point for point in points if point[0] >= T]
            if not lower:
                point_T, point_Cp = points[0]
                return point_Cp, f"clamped below table range to {point_T:g} K"
            if not upper:
                point_T, point_Cp = points[-1]
                return point_Cp, f"clamped above table range to {point_T:g} K"
            T1, Cp1 = lower[-1]
            T2, Cp2 = upper[0]
            if abs(T2 - T1) < 1e-12:
                return Cp1, f"exact point at {T1:g} K"
            value = Cp1 + (Cp2 - Cp1) * (T - T1) / (T2 - T1)
            return value, f"linear interpolation between {T1:g} K and {T2:g} K"


        @staticmethod
        def _shomate_cp_integral_value(coefficients: list[float], T1: float, T2: float) -> float:
            t1 = T1 / 1000.0
            t2 = T2 / 1000.0
            A, B, C, D, E = coefficients
            total = A * (t2 - t1)
            total += B * (t2 * t2 - t1 * t1) / 2.0
            total += C * (t2**3 - t1**3) / 3.0
            total += D * (t2**4 - t1**4) / 4.0
            total += -E * (1.0 / t2 - 1.0 / t1)
            return total


        def _integrate_nist_cp_tables(
            self,
            tables: Dict[str, list[list[float]]],
            T1: float,
            T2: float,
            phase: str,
        ) -> Optional[HeatCapacityIntegralLookup]:
            phase_key = phase.strip().lower().replace('-', '_')
            if phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                table_phase = 'gas'
            elif phase_key == 'liquid':
                table_phase = 'liquid'
            elif phase_key in {'solid', 'crystal', 'crystalline'}:
                table_phase = 'solid'
            else:
                return None

            raw_points = tables.get(table_phase) or []
            points = sorted((float(row[0]), float(row[1])) for row in raw_points if len(row) >= 2)
            if table_phase == 'liquid':
                kernel = self._kernel_from_nist_liquid_source(tables)
                if kernel is None:
                    return None
                value = kernel.delta_h(T1, T2) / 1000.0
                return HeatCapacityIntegralLookup(
                    value=value,
                    phase=table_phase,
                    T1=T1,
                    T2=T2,
                    method=f'{kernel.method}_integral',
                    quality=min(kernel.quality_at(T1), kernel.quality_at(T2)),
                    notes=f'Analytic integral from normalized online liquid Cp kernel; {kernel.notes}',
                )
            if table_phase == 'solid':
                kernel = self._kernel_from_nist_solid_source(tables)
                if kernel is None:
                    return None
                value = kernel.delta_h(T1, T2) / 1000.0
                return HeatCapacityIntegralLookup(
                    value=value,
                    phase=table_phase,
                    T1=T1,
                    T2=T2,
                    method=f'{kernel.method}_integral',
                    quality=min(kernel.quality_at(T1), kernel.quality_at(T2)),
                    notes=f'Analytic integral from normalized online solid Cp kernel; {kernel.notes}',
                )

            if not points:
                return None

            lo = min(T1, T2)
            hi = max(T1, T2)
            if len(points) >= 10:
                fit = self._fit_shomate_cp(points)
                if fit is not None:
                    coefficients, kept_points, excluded, mape, max_error = fit
                    Tmin = min(point_T for point_T, _ in kept_points)
                    Tmax = max(point_T for point_T, _ in kept_points)
                    if Tmin <= lo + 1e-9 and hi <= Tmax + 1e-9:
                        value = self._shomate_cp_integral_value(coefficients, T1, T2)
                        return HeatCapacityIntegralLookup(
                            value=value,
                            phase=table_phase,
                            T1=T1,
                            T2=T2,
                            method='nist_shomate_gas_cp_integral',
                            quality=0.88,
                            notes=(
                                f"Analytic integral of Shomate-style gas Cp fit over {Tmin:g}-{Tmax:g} K; "
                                f"MAPE {mape:.2f}%, max error {max_error:.2f}%; "
                                f"excluded {len(excluded)}/{len(points)} point(s)"
                            ),
                        )
                return None

            if 4 <= len(points) <= 9:
                linear = self._linear_cp_fit(points)
                if linear is None:
                    return None
                intercept, slope, mape, max_error = linear
                value = (
                    intercept * (T2 - T1)
                    + slope * (T2 * T2 - T1 * T1) / 2.0
                ) / 1000.0
                extrapolation = ''
                if lo < points[0][0]:
                    extrapolation += '; extrapolated below table range'
                if hi > points[-1][0]:
                    extrapolation += '; extrapolated above table range'
                return HeatCapacityIntegralLookup(
                    value=value,
                    phase=table_phase,
                    T1=T1,
                    T2=T2,
                    method='nist_linear_gas_cp_integral',
                    quality=0.76,
                    notes=(
                        f"Analytic integral of linear gas Cp fit from {len(points)} NIST point(s), "
                        f"table range {points[0][0]:g}-{points[-1][0]:g} K; "
                        f"MAPE {mape:.2f}%, max error {max_error:.2f}%{extrapolation}"
                    ),
                )
            return None


        def _evaluate_nist_cp_tables(
            self,
            tables: Dict[str, list[list[float]]],
            T: float,
            phase: str,
        ) -> Optional[HeatCapacityLookup]:
            phase_key = phase.strip().lower().replace('-', '_')
            if phase_key in {'gas', 'vapor', 'vapour', 'ideal_gas', 'ideal'}:
                table_phase = 'gas'
            elif phase_key == 'liquid':
                table_phase = 'liquid'
            elif phase_key in {'solid', 'crystal', 'crystalline'}:
                table_phase = 'solid'
            else:
                return None

            raw_points = tables.get(table_phase) or []
            points = sorted((float(row[0]), float(row[1])) for row in raw_points if len(row) >= 2)
            if table_phase == 'liquid':
                kernel = self._kernel_from_nist_liquid_source(tables)
                if kernel is None:
                    return None
                evaluation = kernel.evaluate(T)
                notes = kernel.notes
                if evaluation.range_note:
                    notes = f'{notes}; {evaluation.range_note}' if notes else evaluation.range_note
                return self._cp_lookup_result(
                    evaluation.value,
                    table_phase,
                    T,
                    kernel.method,
                    evaluation.quality,
                    notes,
                )
            if table_phase == 'solid':
                kernel = self._kernel_from_nist_solid_source(tables)
                if kernel is None:
                    return None
                evaluation = kernel.evaluate(T)
                active = kernel.active_kernel(T)
                notes = active.notes
                if evaluation.range_note:
                    notes = f'{notes}; {evaluation.range_note}' if notes else evaluation.range_note
                return self._cp_lookup_result(
                    evaluation.value,
                    table_phase,
                    T,
                    active.method,
                    evaluation.quality,
                    notes,
                )

            if not points:
                return None

            if len(points) >= 10:
                fit = self._fit_shomate_cp(points)
                if fit is not None:
                    coefficients, kept_points, excluded, mape, max_error = fit
                    Tmin = min(point_T for point_T, _ in kept_points)
                    Tmax = max(point_T for point_T, _ in kept_points)
                    if Tmin <= T <= Tmax:
                        value = sum(
                            coefficient * basis
                            for coefficient, basis in zip(coefficients, self._shomate_cp_basis(T))
                        )
                        return self._cp_lookup_result(
                            value,
                            table_phase,
                            T,
                            'nist_shomate_gas_cp_fit',
                            0.88,
                            (
                                f"Shomate-style gas Cp fit over {Tmin:g}-{Tmax:g} K; "
                                f"MAPE {mape:.2f}%, max error {max_error:.2f}%; "
                                f"excluded {len(excluded)}/{len(points)} point(s)"
                            ),
                        )

                value, note = self._interpolate_or_clamp_cp(points, T)
                return self._cp_lookup_result(
                    value,
                    table_phase,
                    T,
                    'nist_tabulated_gas_cp',
                    0.78,
                    f"fit unavailable; {note}",
                )

            if 4 <= len(points) <= 9:
                linear = self._linear_cp_fit(points)
                if linear is not None:
                    intercept, slope, mape, max_error = linear
                    value = intercept + slope * T
                    Tmin = points[0][0]
                    Tmax = points[-1][0]
                    if 1.0 <= value <= 1000.0:
                        extrapolation = ''
                        if T < Tmin:
                            extrapolation = '; extrapolated below table range'
                        elif T > Tmax:
                            extrapolation = '; extrapolated above table range'
                        return self._cp_lookup_result(
                            value,
                            table_phase,
                            T,
                            'nist_linear_gas_cp_fit',
                            0.76,
                            (
                                f"linear gas Cp fit from {len(points)} point(s), table range {Tmin:g}-{Tmax:g} K; "
                                f"MAPE {mape:.2f}%, max error {max_error:.2f}%{extrapolation}"
                            ),
                        )

            value, note = self._interpolate_or_clamp_cp(points, T)
            return self._cp_lookup_result(
                value,
                table_phase,
                T,
                'nist_tabulated_gas_cp',
                0.72,
                note,
            )
