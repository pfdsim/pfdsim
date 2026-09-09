import sqlite3
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from .common import *
from .cache_paths import SATURATION_PROPERTIES_CACHE_PATH
from .cache_expiration import runtime_cache_row_is_fresh
from .coolprop import (
    COOLPROP_PROPERTY_QUALITY,
    coolprop_props_si,
    coolprop_reference_for,
)
from .organic_classification import classify_strict_molecular_organic
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..psrk_parameters import psrk_unstarred_critical_values
else:
    from psrk_parameters import psrk_unstarred_critical_values


EFFECTIVE_CRITICALS_DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'effective_criticals.sqlite'
ACS_JCED_5B00571_TABLE1_PATH = Path(__file__).resolve().parent.parent / 'data' / 'acs_jced_5b00571_table1.json'
CRITICAL_PROPERTIES_CACHE_VERSION = 17
CRITICAL_COMPRESSIBILITY_MINIMUM = 0.1
CRITICAL_COMPRESSIBILITY_MAXIMUM = 0.6
ORGANIC_CRITICAL_PRESSURE_MAXIMUM_BAR = 500.0
INORGANIC_CRITICAL_PRESSURE_MAXIMUM_BAR = 2000.0
EFFECTIVE_CRITICAL_ONLINE_REPLACEABLE_QUALITY = 0.90
EFFECTIVE_CRITICAL_PSRK_PRIORITY_QUALITY = 0.95
PSAT_OMEGA_QUALITY_PENALTY = 0.02
PC_FALLBACK_TREF_K = 373.15
PC_FALLBACK_RING_A = 1.198
PC_FALLBACK_RING_B = -0.154
PC_FALLBACK_TB_EXPONENT = 0.8
TB_FORMULA_HBD_COEFFICIENTS = {
    'A': 166.312,
    'H': -2.95305,
    'C': 67.7668,
    'C_exp': 0.685653,
    'N': 38.8193,
    'O': 23.4180,
    'X': 13.5882,
    'other': 27.9136,
    'HBD': 46.7294,
}
TB_FORMULA_NO_HBD_COEFFICIENTS = {
    **TB_FORMULA_HBD_COEFFICIENTS,
    'N': 53.079496,
    'O': 32.71472,
    'HBD': 0.0,
}
TB_HALOGEN_WEIGHTS = {'F': 1.0, 'Cl': 2.0, 'Br': 4.0, 'I': 6.0}
TB_MW_FALLBACK_COEFFICIENT = 64.29202662
TB_MW_FALLBACK_EXPONENT = 0.39694617
VC_FORMULA_FALLBACK_COEFFICIENTS = {
    'A': 8.994812794652804,
    'H': 8.941516207451158,
    'C': 39.66982548990393,
    'O': 24.73675241515339,
    'N': 35.70581344450935,
    'F': 21.928124550659955,
    'Si': 98.3999636419683,
    'SClP': 58.232883957771875,
    'other': 78.09338394554524,
    'large_ring': -40.251547492335234,
}


class CriticalPropertiesMixin:
        CRITICAL_PROPERTIES_CACHE_VERSION = CRITICAL_PROPERTIES_CACHE_VERSION
        SATURATION_PROPERTIES_CACHE_PATH = SATURATION_PROPERTIES_CACHE_PATH

        def resolve_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any] = None,
            allow_online: bool = True,
            allow_estimation: bool = True,
        ) -> Dict[str, PropertyResolutionResult]:
            """
            Resolve critical properties (Tc, Pc, Vc, Zc, omega).

            Args:
                symbol: Chemical symbol
                props: Optional dict with any known properties

            Returns:
                Dict mapping property names to resolution results
            """
            props = self._coerce_props(symbol, props, allow_online=allow_online)
            props = self._critical_props_with_resolved_boiling_point(
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
            )
            component_key, _cas, _name = self._critical_cache_component_identity(
                symbol,
                props,
            )
            fingerprint = self._critical_cache_input_fingerprint(
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
            )
            cache_key = (
                int(self.CRITICAL_PROPERTIES_CACHE_VERSION),
                component_key,
                fingerprint,
            )
            memory_cache = getattr(self, '_resolved_critical_properties_cache', None)
            if memory_cache is None:
                memory_cache = {}
                self._resolved_critical_properties_cache = memory_cache
            cached = memory_cache.get(cache_key)
            if cached is not None:
                return self._copy_critical_results(cached)

            persistent = self._load_persistent_critical_properties(
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
                component_key=component_key,
                fingerprint=fingerprint,
            )
            if persistent is not None:
                memory_cache[cache_key] = persistent
                return self._copy_critical_results(persistent)

            with self._online_attempt_scope(allow_online) as online_attempt:
                results = self._resolve_critical_properties_uncached(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=allow_estimation,
                )
            if not allow_online and (
                props.get('critical_properties_unavailable')
                or all(
                    result is not None
                    and result.value is not None
                    and not self._result_is_soft(result)
                    for result in results.values()
                )
            ):
                online_attempt.record(OnlineAttemptState.NOT_NEEDED)
            if self._online_attempt_is_persistable(
                allow_online,
                online_attempt.state,
            ):
                has_pfd_override = (
                    any(
                        self._is_pfd_component_override(props, name)
                        for name in ('Tc', 'Pc', 'Vc', 'Zc', 'omega')
                    )
                    or self._is_pfd_correlation_override(props, 'Psat')
                )
                if not has_pfd_override:
                    self._store_persistent_critical_properties(
                        symbol,
                        props,
                        results,
                        allow_online=allow_online,
                        allow_estimation=allow_estimation,
                        online_attempt_state=online_attempt.state,
                        component_key=component_key,
                        fingerprint=fingerprint,
                    )
                memory_cache[cache_key] = self._copy_critical_results(results)
            return results


        def _critical_props_with_resolved_boiling_point(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> Dict[str, Any]:
            """Resolve a missing/soft Tb before it is consumed by critical estimates."""
            stored_tb = self._source_result_for_value(props, 'Tb', units='K')
            if stored_tb is not None and not self._result_is_soft(stored_tb):
                return props

            working = dict(props)
            sources = dict(working.get('property_sources') or {})
            working['property_sources'] = sources
            correlations = working.get('property_correlations') or {}
            psat_correlation = (
                correlations.get('Psat') or correlations.get('psat')
                if isinstance(correlations, Mapping)
                else None
            )
            has_pfd_psat_override = bool(
                isinstance(psat_correlation, Mapping)
                and psat_correlation.get('_pfd_override')
            )

            if not has_pfd_psat_override:
                triple = self.resolve_triple_point(
                    symbol,
                    working,
                    # ChemicalDatabase performs the full ordered online pass.
                    # Sparse direct critical calls keep this prerequisite local
                    # so they do not fan out into several network lookups.
                    allow_online=False,
                )
                for field_name in ('Tt', 'Pt'):
                    result = triple.get(field_name)
                    if result is None or result.value is None:
                        continue
                    working[field_name] = result.value
                    sources[field_name] = {
                        'source': result.source,
                        'method': result.method,
                        'quality': result.quality,
                        'notes': result.notes,
                    }

            boiling = self.resolve_boiling_point(
                symbol,
                working,
                allow_online=False,
                allow_estimation=allow_estimation,
            )
            if boiling.value is not None:
                working['Tb'] = boiling.value
                sources['Tb'] = {
                    'source': boiling.source,
                    'method': boiling.method,
                    'quality': boiling.quality,
                    'notes': boiling.notes,
                }
            elif boiling.method in {
                'no_normal_boiling_point_at_1atm',
                'invalid_normal_boiling_point_below_triple_point',
            }:
                working.pop('Tb', None)
                sources.pop('Tb', None)
            return working


        def _resolve_critical_properties_uncached(
            self,
            symbol: str,
            props: Dict[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> Dict[str, PropertyResolutionResult]:
            """Run the established provider and arbitration chain unchanged."""
            if props.get('critical_properties_unavailable'):
                return {
                    prop_name: PropertyResolutionResult(
                        value=None,
                        source='missing',
                        method='critical_properties_unavailable',
                        quality=0.0,
                        notes='Critical properties are intentionally unavailable for this component',
                    )
                    for prop_name in ['Tc', 'Pc', 'Vc', 'Zc', 'omega']
                }

            results = {}
            textbook_props = None
            for candidate in self._identifier_candidates(symbol, props):
                textbook_props = self._get_textbook_entry(candidate)
                if textbook_props:
                    break
            acs_jced_props = self._get_acs_jced_critical_properties(symbol, props)
            perry_props = self._get_perry_critical_properties(symbol, props)
            psrk_props = self._get_psrk_critical_properties(symbol, props)
            effective_props = self._get_effective_critical_properties(symbol, props)
            coolprop_props = self._coolprop_critical_properties(symbol, props)
            online_phase_props = None
            tried_online = False

            def candidate_admissible(
                prop_name: str,
                candidate: Optional[PropertyResolutionResult],
            ) -> bool:
                if candidate is None or candidate.value is None:
                    return False
                if self._is_pfd_component_override(props, prop_name):
                    return True
                if (
                    str(candidate.method) == 'effective_critical'
                    and self._critical_formula_is_metal_or_salt(props)
                ):
                    return True
                if prop_name == 'Pc':
                    return self._critical_pressure_is_admissible(
                        props,
                        candidate.value,
                    )
                if prop_name == 'Zc':
                    return self._critical_compressibility_is_admissible(
                        candidate.value,
                    )
                return True

            for prop_name in ['Tc', 'Pc', 'Vc', 'Zc', 'omega']:
                source_metadata = (
                    ((props or {}).get('property_sources') or {}).get(prop_name)
                    or {}
                )
                replacement_locked = source_metadata.get('replaceable') is False
                # Check if already provided
                if prop_name in props and props[prop_name] is not None:
                    provided_result = self._source_result_for_value(
                        props,
                        prop_name,
                        units={'Tc': 'K', 'Pc': 'bar', 'Vc': 'cm^3/mol'}.get(prop_name, ''),
                    )
                    if (
                        not self._critical_source_is_estimated(props, prop_name)
                        and candidate_admissible(prop_name, provided_result)
                    ):
                        results[prop_name] = PropertyResolutionResult(
                            value=provided_result.value,
                            source=provided_result.source,
                            method=provided_result.method,
                            quality=provided_result.quality,
                            notes=provided_result.notes,
                        )
                        if self._is_pfd_component_override(props, prop_name):
                            continue
                    if (
                        provided_result
                        and self._critical_source_is_estimated(props, prop_name)
                        and candidate_admissible(prop_name, provided_result)
                    ):
                        results[prop_name] = provided_result
                    if replacement_locked and prop_name in results:
                        continue

                if prop_name == 'Zc' and not self._is_pfd_component_override(props, 'Zc'):
                    primitive_results = [
                        self._critical_input_result(props, results, name)
                        for name in ('Tc', 'Pc', 'Vc')
                    ]
                    coupled_provider_changed = (
                        any(
                            item and str(item.method).startswith('coolprop_')
                            for item in primitive_results
                        )
                        or any(
                            self._is_pfd_component_override(props, name)
                            for name in ('Tc', 'Pc', 'Vc')
                        )
                    )
                    if coupled_provider_changed:
                        identity = self._critical_zc_identity_result(props, results)
                        if identity:
                            results[prop_name] = identity
                            continue

                if coolprop_props and prop_name in coolprop_props:
                    candidate = coolprop_props[prop_name]
                    if not candidate_admissible(prop_name, candidate):
                        candidate = None
                    if candidate is None:
                        pass
                    elif not self._is_pfd_component_override(props, prop_name):
                        results[prop_name] = candidate
                        continue
                    elif self._critical_result_can_replace(results.get(prop_name), candidate):
                        results[prop_name] = candidate
                        continue

                if acs_jced_props and prop_name in acs_jced_props:
                    candidate = acs_jced_props[prop_name]
                    if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                        results[prop_name] = candidate
                        continue

                if perry_props and prop_name in perry_props:
                    item = perry_props[prop_name]
                    candidate = PropertyResolutionResult(
                        value=item.value,
                        source='local',
                        method=item.method,
                        quality=0.98,
                        notes=f"{item.source}; units {item.units}"
                    )
                    if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                        results[prop_name] = candidate
                        continue

                if textbook_props and textbook_props.get(prop_name) is not None:
                    candidate = PropertyResolutionResult(
                        value=textbook_props[prop_name],
                        source='textbook',
                        method='Smith8 Appendix B',
                        quality=0.98
                    )
                    if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                        results[prop_name] = candidate
                        continue

                effective_result = None
                if effective_props and prop_name in effective_props:
                    effective_result = self._effective_critical_result(
                        effective_props,
                        prop_name,
                    )
                    if (
                        effective_result.quality
                        >= EFFECTIVE_CRITICAL_PSRK_PRIORITY_QUALITY
                        and candidate_admissible(prop_name, effective_result)
                        and self._critical_result_can_replace(
                            results.get(prop_name), effective_result
                        )
                    ):
                        results[prop_name] = effective_result
                        continue

                if psrk_props and prop_name in psrk_props:
                    candidate = psrk_props[prop_name]
                    if candidate_admissible(
                        prop_name, candidate
                    ) and self._critical_result_can_replace(
                        results.get(prop_name), candidate
                    ):
                        results[prop_name] = candidate
                        continue

                if effective_result is not None:
                    if (
                        effective_result.quality
                        >= EFFECTIVE_CRITICAL_ONLINE_REPLACEABLE_QUALITY
                        and candidate_admissible(prop_name, effective_result)
                        and self._critical_result_can_replace(results.get(prop_name), effective_result)
                    ):
                        results[prop_name] = effective_result
                        continue

                # Use the shared phase-change path for online criticals. It applies
                # strict unit parsing and lets curated NIST values replace the
                # more heterogeneous PubChem fallback.
                if allow_online and not tried_online:
                    tried_online = True
                    try:
                        online_phase_props = self._fetch_phase_change_online(symbol, props)
                    except LookupError:
                        online_phase_props = None

                if online_phase_props and prop_name in online_phase_props:
                    source_map = online_phase_props.get('_sources', {})
                    quality_map = online_phase_props.get('_qualities', {})
                    notes_map = online_phase_props.get('_notes', {})
                    method = source_map.get(prop_name, 'nist_phase_change')
                    candidate = PropertyResolutionResult(
                        value=online_phase_props[prop_name],
                        source='online',
                        method=method,
                        quality=self._clamp_quality(
                            quality_map.get(prop_name),
                            0.95 if method == 'nist_phase_change' else 0.88,
                        ),
                        notes=notes_map.get(
                            prop_name,
                            'Strict unit-aware online critical-property parsing',
                        ),
                    )
                    if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                        results[prop_name] = candidate
                        continue

                if effective_result and candidate_admissible(prop_name, effective_result) and self._critical_result_can_replace(results.get(prop_name), effective_result):
                    results[prop_name] = effective_result

                if prop_name == 'Zc':
                    identity = self._critical_zc_identity_result(props, results)
                    if identity and candidate_admissible(prop_name, identity) and self._critical_result_can_replace(results.get(prop_name), identity):
                        results[prop_name] = identity
                        continue

                if allow_estimation and prop_name in {'Tc', 'Pc', 'Vc'}:
                    nannoolal = self._estimate_nannoolal_critical_property(
                        props,
                        results,
                        prop_name,
                        allow_online=allow_online,
                    )
                    if nannoolal and candidate_admissible(prop_name, nannoolal) and self._critical_result_can_replace(results.get(prop_name), nannoolal):
                        results[prop_name] = nannoolal
                        continue

                # Final empirical fallbacks. Keep these in the resolver so
                # direct resolver calls and ChemicalProperties hydration agree.
                if allow_estimation and prop_name == 'Tc':
                    tb_result = self._critical_tb_input_result(props)
                    if tb_result and tb_result.value is not None:
                        quality = min(0.8 * self._result_quality(tb_result, 0.0), 0.60)
                        candidate = PropertyResolutionResult(
                            value=1.5 * float(tb_result.value),
                            source='estimated',
                            method='guldberg_rule',
                            quality=quality,
                            notes=(
                                'Estimated from molecular-weight boiling-point estimate'
                                if tb_result.method == 'mw_correlation'
                                else 'Estimated from Tb'
                            ),
                        )
                        if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                            results[prop_name] = candidate
                            continue

                if allow_estimation and prop_name == 'Pc':
                    pc_estimate = self._estimate_pc_from_formula_tb(props, allow_online=allow_online)
                    if pc_estimate:
                        value, notes = pc_estimate
                        tb_result = self._critical_tb_input_result(props)
                        quality = min(self._result_quality(tb_result, 0.55), 0.55)
                        candidate = PropertyResolutionResult(
                            value=value,
                            source='estimated',
                            method='atom_count_ring_tb_pc',
                            quality=quality,
                            notes=notes,
                        )
                        if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                            results[prop_name] = candidate
                            continue

                if allow_estimation and prop_name == 'Vc':
                    vc_estimate = self._estimate_vc_from_formula_structure(props, allow_online=allow_online)
                    if vc_estimate:
                        value, inputs, ring_available, notes = vc_estimate
                        quality = min(self._combine_quality(inputs, method_factor=0.8), 0.8)
                        if not ring_available:
                            quality = max(0.0, quality - 0.05)
                        candidate = PropertyResolutionResult(
                            value=value,
                            source='estimated',
                            method='atom_count_large_ring_vc',
                            quality=quality,
                            notes=notes,
                        )
                        if candidate_admissible(prop_name, candidate) and self._critical_result_can_replace(results.get(prop_name), candidate):
                            results[prop_name] = candidate
                            continue

                if allow_estimation and prop_name == 'omega':
                    Tc_result = self._critical_input_result(props, results, 'Tc')
                    Pc_result = self._critical_input_result(props, results, 'Pc')
                    Tc = Tc_result.value if Tc_result else None
                    Pc = Pc_result.value if Pc_result else None
                    Tb_result = self._critical_tb_input_result(props)
                    Tb = Tb_result.value if Tb_result else None

                    if Tc and Pc and Tb:
                        omega_est = self._estimate_omega(Tc, Pc, Tb)
                        if omega_est is not None:
                            inputs = [item for item in (Tc_result, Pc_result, Tb_result) if item]
                            quality_factor, quality_note = self._lee_kesler_quality_factor(
                                props,
                                allow_online=allow_online,
                            )
                            estimated_result = PropertyResolutionResult(
                                value=omega_est,
                                source='calculated',
                                method='lee_kesler',
                                quality=self._combine_quality(inputs, method_factor=quality_factor),
                                notes=f'Estimated from Tc, Pc, Tb; {quality_note}'
                            )
                            if self._critical_result_can_replace(results.get(prop_name), estimated_result):
                                results[prop_name] = estimated_result
                                continue

                # Property not resolved
                results.setdefault(
                    prop_name,
                    PropertyResolutionResult(
                        value=None,
                        source='missing',
                        method='none',
                        quality=0.0,
                        notes=f'{prop_name} not available'
                    ),
                )

            existing_omega = results.get('omega')
            if (
                not self._is_pfd_component_override(props, 'omega')
                and self._result_quality(existing_omega, 0.0) < 0.98
            ):
                psat_omega = self._independent_psat_omega_result(
                    symbol,
                    props,
                    results,
                )
                if (
                    psat_omega is not None
                    and psat_omega.value is not None
                    and (
                        existing_omega is None
                        or existing_omega.value is None
                        or self._result_quality(psat_omega, 0.0)
                        > self._result_quality(existing_omega, 0.0) + 1.0e-12
                    )
                ):
                    results['omega'] = psat_omega
            return results


        def _independent_psat_omega_result(
            self,
            symbol: str,
            props: Dict[str, Any],
            results: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            """Recover omega at Tr=0.7 from selected omega-independent Psat."""
            Tc_result = self._critical_input_result(props, results, 'Tc')
            Pc_result = self._critical_input_result(props, results, 'Pc')
            if (
                Tc_result is None
                or Pc_result is None
                or Tc_result.value is None
                or Pc_result.value is None
            ):
                return None
            correlations = props.get('property_correlations') or {}
            psat_correlation = (
                correlations.get('Psat') or correlations.get('psat')
                if isinstance(correlations, Mapping)
                else None
            )
            if (
                isinstance(psat_correlation, Mapping)
                and psat_correlation.get('_pfd_override')
            ):
                return None
            if any(
                self._is_pfd_component_override(props, field_name)
                for field_name in ('Tc', 'Pc')
            ):
                return None
            if any(
                str(item.method or '').strip().lower() == 'effective_critical'
                for item in (Tc_result, Pc_result)
            ):
                return None
            try:
                Tc = float(Tc_result.value)
                Pc_bar = float(Pc_result.value)
            except (TypeError, ValueError):
                return None
            if (
                not math.isfinite(Tc)
                or Tc <= 0.0
                or not math.isfinite(Pc_bar)
                or Pc_bar <= 0.0
            ):
                return None

            from .vapor_pressure_adapter import (
                DIRECT_PSAT_INPUT_METHODS,
                PsatCanonicalizationAdapter,
            )
            from .vapor_pressure_canonical import (
                PsatCanonicalizationError,
                PsatHandoffCoordinator,
            )

            component = dict(props)
            sources = dict(component.get('property_sources') or {})
            component['property_sources'] = sources
            for field_name, item in (('Tc', Tc_result), ('Pc', Pc_result)):
                component[field_name] = item.value
                sources[field_name] = {
                    'source': item.source,
                    'method': item.method,
                    'quality': item.quality,
                    'notes': item.notes,
                }
            component.pop('omega', None)
            sources.pop('omega', None)
            component.setdefault('symbol', str(symbol))
            component.setdefault('name', str(symbol))

            target_temperature = 0.7 * Tc
            adapter = PsatCanonicalizationAdapter(
                component,
                input_methods=DIRECT_PSAT_INPUT_METHODS,
            )
            try:
                direct = adapter.collect_inputs(
                    T_min=max(1.0, 0.15 * Tc),
                    T_critical=Tc,
                    P_critical_bar=Pc_bar,
                    T_boiling=component.get('Tb'),
                )
            except (ArithmeticError, LookupError, TypeError, ValueError):
                return None

            selected_source = selected_method = None
            selected_quality = None
            selected_dependencies = None
            selected_tb_validation_status = None
            ln_pressure = None
            if direct.canonical_override is not None:
                curve = direct.canonical_override
                if not curve.T_min <= target_temperature <= curve.T_critical:
                    return None
                provenance = next(
                    (
                        item
                        for item in curve.provenance
                        if item.T_min <= target_temperature <= item.T_max
                    ),
                    None,
                )
                if provenance is None:
                    return None
                dependencies = provenance.metadata.get('property_dependencies')
                if dependencies is None or 'omega' in set(dependencies):
                    return None
                try:
                    ln_pressure = float(curve.ln_pressure(target_temperature))
                except (ArithmeticError, TypeError, ValueError):
                    return None
                selected_source = provenance.source
                selected_method = provenance.method
                selected_quality = provenance.quality
                selected_dependencies = tuple(dependencies)
                selected_tb_validation_status = provenance.metadata.get(
                    'tb_validation_status'
                )
            else:
                eligible = tuple(
                    segment
                    for segment in direct.segments
                    if (
                        segment.metadata.get('property_dependencies') is not None
                        and 'omega' not in set(
                            segment.metadata.get('property_dependencies') or ()
                        )
                    )
                )
                if not eligible:
                    return None
                try:
                    coordinated = PsatHandoffCoordinator(
                        min(segment.T_min for segment in eligible),
                        max(segment.T_max for segment in eligible),
                    ).coordinate(eligible)
                except PsatCanonicalizationError:
                    return None
                selected = coordinated.assembly.slice_at(target_temperature)
                if selected is None:
                    return None
                dependencies = selected.segment.metadata.get(
                    'property_dependencies'
                )
                if dependencies is None or 'omega' in set(dependencies):
                    return None
                try:
                    ln_pressure = float(selected.ln_pressure(target_temperature))
                except (ArithmeticError, TypeError, ValueError):
                    return None
                selected_source = selected.segment.source
                selected_method = selected.segment.method
                selected_quality = selected.segment.quality
                selected_dependencies = tuple(dependencies)
                selected_tb_validation_status = selected.segment.metadata.get(
                    'tb_validation_status'
                )

            # A soft critical pair makes the reduced-temperature coordinate
            # uncertain.  Do not compound that uncertainty with a standalone
            # Psat segment that has not independently reproduced a hard normal
            # boiling point.  Hard Tc/Pc may still recover omega without Tb.
            if (
                any(self._result_is_soft(item) for item in (Tc_result, Pc_result))
                and selected_tb_validation_status != 'hard_tb_validated'
            ):
                return None

            if ln_pressure is None or not math.isfinite(ln_pressure):
                return None
            omega = -(ln_pressure - math.log(Pc_bar)) / math.log(10.0) - 1.0
            if not math.isfinite(omega) or not -0.5 <= omega <= 2.0:
                return None
            input_quality = min(
                self._result_quality(Tc_result, 0.0),
                self._result_quality(Pc_result, 0.0),
                self._clamp_quality(selected_quality, 0.0),
            )
            quality = max(0.0, input_quality - PSAT_OMEGA_QUALITY_PENALTY)
            if quality <= 0.0:
                return None
            pressure_bar = math.exp(ln_pressure)
            return PropertyResolutionResult(
                value=omega,
                source='calculated',
                method='psat_definition_at_Tr_0_7',
                quality=quality,
                notes=(
                    f'Pitzer acentric-factor definition from selected direct '
                    f'Psat source {selected_source}/{selected_method} at '
                    f'T=0.7*Tc={target_temperature:g} K; '
                    f'Psat={pressure_bar:g} bar, Pc={Pc_bar:g} bar; '
                    f'Psat property dependencies={selected_dependencies}; '
                    f'Tb validation status={selected_tb_validation_status}; '
                    f'input quality={input_quality:g}, flat Psat omega '
                    f'quality penalty={PSAT_OMEGA_QUALITY_PENALTY:g}'
                ),
            )


        @staticmethod
        def _copy_critical_results(
            results: Mapping[str, PropertyResolutionResult],
        ) -> Dict[str, PropertyResolutionResult]:
            return {
                str(name): PropertyResolutionResult(
                    value=result.value,
                    source=str(result.source),
                    method=str(result.method),
                    quality=float(result.quality),
                    notes=str(result.notes or ''),
                )
                for name, result in results.items()
            }


        @staticmethod
        def _critical_cache_component_identity(
            symbol: str,
            props: Mapping[str, Any],
        ) -> tuple[str, str, str]:
            cas = str(props.get('CAS') or props.get('cas') or '').strip()
            name = str(props.get('name') or symbol).strip()
            if cas:
                return f"cas:{cas.lower()}", cas, name
            normalized = ''.join(
                character.lower()
                for character in str(symbol)
                if character.isalnum()
            )
            return f"symbol:{normalized or str(symbol).lower()}", '', name


        @staticmethod
        def _critical_cache_json_default(value: Any):
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, (set, frozenset)):
                return sorted(value, key=repr)
            item = getattr(value, 'item', None)
            if callable(item):
                try:
                    return item()
                except Exception:
                    pass
            return repr(value)


        @classmethod
        def _critical_cache_json(cls, value: Any) -> str:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(',', ':'),
                default=cls._critical_cache_json_default,
            )


        @classmethod
        def _critical_cache_input_metadata(
            cls,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> Dict[str, Any]:
            relevant_fields = (
                'CAS', 'cas', 'name', 'symbol', 'formula', 'Formula',
                'smiles', 'SMILES', 'source', 'MW', 'Tb',
                'Tc', 'Pc', 'Vc', 'Zc', 'omega',
                'critical_properties_unavailable',
                'antoine_A', 'antoine_B', 'antoine_C',
                'antoine_Tmin', 'antoine_Tmax', 'antoine_source',
            )
            relevant_props = {
                field_name: props.get(field_name)
                for field_name in relevant_fields
                if field_name in props
            }
            property_sources = props.get('property_sources')
            if isinstance(property_sources, Mapping):
                source_fields = {
                    'formula', 'smiles', 'MW', 'Tb',
                    'Tc', 'Pc', 'Vc', 'Zc', 'omega',
                }
                relevant_props['property_sources'] = {
                    str(field_name): metadata
                    for field_name, metadata in property_sources.items()
                    if str(field_name) in source_fields
                }
            property_correlations = props.get('property_correlations')
            if isinstance(property_correlations, Mapping):
                psat_correlation = (
                    property_correlations.get('Psat')
                    or property_correlations.get('psat')
                )
                if isinstance(psat_correlation, Mapping):
                    relevant_props['property_correlations'] = {
                        'Psat': dict(psat_correlation),
                    }
            return {
                'resolver_contract': 'resolved_critical_properties_v4',
                'symbol_argument': str(symbol),
                'allow_online': bool(allow_online),
                'allow_estimation': bool(allow_estimation),
                'properties': relevant_props,
                'units': {
                    'Tc': 'K',
                    'Pc': 'bar',
                    'Vc': 'cm^3/mol',
                    'Zc': 'dimensionless',
                    'omega': 'dimensionless',
                },
                'provider_order': [
                    'provided',
                    'CoolProp',
                    'ACS JCED 2015/IUPAC',
                    'Perry 9th',
                    'Smith8 textbook',
                    'effective criticals at quality >=0.95',
                    'PSRK 2005 unstarred Tc/Pc',
                    'other qualifying effective criticals',
                    'online phase change',
                    'derived identities',
                    'Nannoolal and empirical estimation',
                    'final omega quality arbitration with selected '
                    'independent Psat at Tr=0.7',
                ],
            }


        @classmethod
        def _critical_cache_input_fingerprint(
            cls,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
        ) -> str:
            payload = cls._critical_cache_json(
                cls._critical_cache_input_metadata(
                    symbol,
                    props,
                    allow_online=allow_online,
                    allow_estimation=allow_estimation,
                )
            )
            return hashlib.sha256(payload.encode('utf-8')).hexdigest()


        def _ensure_critical_properties_cache_schema(self, connection) -> None:
            property_columns = []
            for property_name in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                property_columns.extend([
                    f'{property_name}_value REAL',
                    f'{property_name}_source TEXT NOT NULL',
                    f'{property_name}_method TEXT NOT NULL',
                    f'{property_name}_quality REAL NOT NULL',
                    f'{property_name}_notes TEXT NOT NULL',
                ])
            connection.execute('PRAGMA busy_timeout = 30000')
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS resolved_critical_properties_cache (
                    cache_version INTEGER NOT NULL,
                    component_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cas TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    allow_online INTEGER NOT NULL,
                    allow_estimation INTEGER NOT NULL,
                    online_attempt_state TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    {', '.join(property_columns)},
                    results_json TEXT NOT NULL,
                    input_metadata_json TEXT NOT NULL,
                    PRIMARY KEY (
                        cache_version,
                        component_key,
                        input_fingerprint
                    )
                )
                """
            )
            existing_columns = {
                str(row[1])
                for row in connection.execute(
                    'PRAGMA table_info(resolved_critical_properties_cache)'
                )
            }
            if 'online_attempt_state' not in existing_columns:
                connection.execute(
                    'ALTER TABLE resolved_critical_properties_cache '
                    "ADD COLUMN online_attempt_state TEXT NOT NULL "
                    "DEFAULT 'not_attempted'"
                )
            for suffix, column in (
                ('cas', 'cas'),
                ('name', 'component_name'),
                ('symbol', 'symbol'),
                ('updated', 'updated_at_utc'),
            ):
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS
                        resolved_critical_properties_cache_{suffix}_idx
                    ON resolved_critical_properties_cache
                        (cache_version, {column})
                    """
                )


        def initialize_critical_properties_disk_cache(self) -> Path:
            """Create the persistent resolved-critical cache and return its path."""
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                connection.execute('PRAGMA journal_mode = WAL')
                self._ensure_critical_properties_cache_schema(connection)
                connection.execute(
                    f'PRAGMA user_version = '
                    f'{int(self.CRITICAL_PROPERTIES_CACHE_VERSION)}'
                )
                connection.commit()
            return path


        def _load_persistent_critical_properties(
            self,
            symbol: str,
            props: Mapping[str, Any],
            *,
            allow_online: bool,
            allow_estimation: bool,
            component_key: str,
            fingerprint: str,
        ) -> Optional[Dict[str, PropertyResolutionResult]]:
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            if not path.exists():
                return None
            try:
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    self._ensure_critical_properties_cache_schema(connection)
                    row = connection.execute(
                        """
                        SELECT * FROM resolved_critical_properties_cache
                        WHERE cache_version = ?
                          AND component_key = ?
                          AND input_fingerprint = ?
                        """,
                        (
                            int(self.CRITICAL_PROPERTIES_CACHE_VERSION),
                            component_key,
                            fingerprint,
                        ),
                    ).fetchone()
                if row is None:
                    return None
                if not runtime_cache_row_is_fresh(
                    path,
                    row['updated_at_utc'],
                ):
                    return None
                if not self._online_attempt_is_persistable(
                    bool(row['allow_online']),
                    str(row['online_attempt_state']),
                ):
                    return None
                results_payload = json.loads(row['results_json'])
                input_metadata = json.loads(row['input_metadata_json'])
                if not isinstance(results_payload, dict) or not isinstance(input_metadata, dict):
                    return None
                results = {}
                for property_name in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                    results[property_name] = PropertyResolutionResult(
                        value=(
                            None
                            if row[f'{property_name}_value'] is None
                            else float(row[f'{property_name}_value'])
                        ),
                        source=str(row[f'{property_name}_source']),
                        method=str(row[f'{property_name}_method']),
                        quality=float(row[f'{property_name}_quality']),
                        notes=str(row[f'{property_name}_notes']),
                    )
                return results
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                OSError,
                sqlite3.Error,
            ):
                return None


        def _store_persistent_critical_properties(
            self,
            symbol: str,
            props: Mapping[str, Any],
            results: Mapping[str, PropertyResolutionResult],
            *,
            allow_online: bool,
            allow_estimation: bool,
            online_attempt_state: OnlineAttemptState | str,
            component_key: str,
            fingerprint: str,
        ) -> None:
            component_key_check, cas, component_name = (
                self._critical_cache_component_identity(symbol, props)
            )
            if component_key_check != component_key:
                return
            ordered_names = ('Tc', 'Pc', 'Vc', 'Zc', 'omega')
            normalized = {
                name: results.get(name) or PropertyResolutionResult(
                    value=None,
                    source='missing',
                    method='none',
                    quality=0.0,
                    notes=f'{name} not available',
                )
                for name in ordered_names
            }
            input_metadata = self._critical_cache_input_metadata(
                symbol,
                props,
                allow_online=allow_online,
                allow_estimation=allow_estimation,
            )
            results_json = self._critical_cache_json({
                name: asdict(result)
                for name, result in normalized.items()
            })
            input_metadata_json = self._critical_cache_json(input_metadata)
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ).replace('+00:00', 'Z')
            columns = [
                'cache_version', 'component_key', 'input_fingerprint',
                'symbol', 'cas', 'component_name', 'allow_online',
                'allow_estimation', 'online_attempt_state',
                'created_at_utc', 'updated_at_utc',
            ]
            values = [
                int(self.CRITICAL_PROPERTIES_CACHE_VERSION),
                component_key,
                fingerprint,
                str(symbol),
                cas,
                component_name,
                int(bool(allow_online)),
                int(bool(allow_estimation)),
                OnlineAttemptState(online_attempt_state).value,
                timestamp,
                timestamp,
            ]
            update_columns = [
                'symbol', 'cas', 'component_name', 'allow_online',
                'allow_estimation', 'online_attempt_state', 'updated_at_utc',
            ]
            for name in ordered_names:
                result = normalized[name]
                columns.extend([
                    f'{name}_value', f'{name}_source', f'{name}_method',
                    f'{name}_quality', f'{name}_notes',
                ])
                values.extend([
                    result.value,
                    str(result.source),
                    str(result.method),
                    float(result.quality),
                    str(result.notes or ''),
                ])
                update_columns.extend([
                    f'{name}_value', f'{name}_source', f'{name}_method',
                    f'{name}_quality', f'{name}_notes',
                ])
            columns.extend(['results_json', 'input_metadata_json'])
            values.extend([results_json, input_metadata_json])
            update_columns.extend(['results_json', 'input_metadata_json'])
            placeholders = ', '.join('?' for _ in columns)
            updates = ', '.join(
                f'{column} = excluded.{column}'
                for column in update_columns
            )
            path = Path(self.SATURATION_PROPERTIES_CACHE_PATH)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(path, timeout=30.0)) as connection:
                    self._ensure_critical_properties_cache_schema(connection)
                    connection.execute(
                        f'PRAGMA user_version = '
                        f'{int(self.CRITICAL_PROPERTIES_CACHE_VERSION)}'
                    )
                    connection.execute(
                        f"""
                        INSERT INTO resolved_critical_properties_cache
                            ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT (
                            cache_version, component_key, input_fingerprint
                        ) DO UPDATE SET {updates}
                        """,
                        values,
                    )
                    connection.commit()
            except (OSError, sqlite3.Error, TypeError, ValueError):
                return


        def _coolprop_reference(self, symbol: str, props: Dict[str, Any]):
            cached = getattr(self, '_coolprop_reference_cache', None)
            if cached is None:
                cached = {}
                self._coolprop_reference_cache = cached
            cache_key = (
                str(symbol),
                str(props.get('CAS') or ''),
                str(props.get('cas') or ''),
            )
            if cache_key not in cached:
                cached[cache_key] = coolprop_reference_for(symbol, props)
            return cached[cache_key]


        def _coolprop_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, PropertyResolutionResult]]:
            reference = self._coolprop_reference(symbol, props)
            if reference is None:
                return None

            values = {
                'Tc': coolprop_props_si('Tcrit', reference),
                'Pc': None,
                'Vc': None,
                'omega': coolprop_props_si('acentric', reference),
            }
            pressure_pa = coolprop_props_si('pcrit', reference)
            if pressure_pa is not None and pressure_pa > 0.0:
                values['Pc'] = pressure_pa / 100000.0
            rhomolar = coolprop_props_si('rhomolar_critical', reference)
            if rhomolar is not None and rhomolar > 0.0:
                values['Vc'] = 1.0e6 / rhomolar

            method = f'{reference.method_prefix}_critical'
            notes = (
                f'CoolProp {reference.backend} pure-fluid critical properties '
                f'for {reference.fluid}; Pc units bar, Vc units cm^3/mol'
            )
            results = {}
            for key, value in values.items():
                if value is None or not math.isfinite(float(value)):
                    continue
                results[key] = PropertyResolutionResult(
                    value=float(value),
                    source='local',
                    method=method,
                    quality=COOLPROP_PROPERTY_QUALITY,
                    notes=notes,
                )
            return results or None


        def _get_acs_jced_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, PropertyResolutionResult]]:
            table = self._load_acs_jced_critical_table()
            if not table:
                return None

            entry = None
            for candidate in self._identifier_candidates(symbol, props):
                candidate_text = str(candidate).strip()
                if self._looks_like_cas(candidate_text):
                    entry = table.get(candidate_text)
                    if entry:
                        break
            if not entry:
                return None

            critical = entry.get('critical_properties') or {}
            results = {}

            def uncertainty_note(label: str, uncertainty_key: str, units: str) -> str:
                uncertainty = critical.get(uncertainty_key)
                pieces = [
                    "ACS JCED 2015/IUPAC critical-property review Table 1 recommended value",
                ]
                if uncertainty is not None:
                    pieces.append(f"{label} uncertainty ±{uncertainty:g} {units}")
                notes = entry.get('extraction_notes') or []
                if notes:
                    pieces.append("; ".join(str(note) for note in notes))
                return "; ".join(pieces)

            if critical.get('Tc_K') is not None:
                results['Tc'] = PropertyResolutionResult(
                    value=critical['Tc_K'],
                    source='local',
                    method='acs_jced_5b00571_table1',
                    quality=0.99,
                    notes=uncertainty_note('Tc', 'Tc_uncertainty_K', 'K'),
                )
            if critical.get('Pc_MPa') is not None:
                results['Pc'] = PropertyResolutionResult(
                    value=critical['Pc_MPa'] * 10.0,
                    source='local',
                    method='acs_jced_5b00571_table1',
                    quality=0.99,
                    notes=uncertainty_note('Pc', 'Pc_uncertainty_MPa', 'MPa')
                    + '; converted from MPa to bar',
                )
            if critical.get('Vc_cm3_mol') is not None:
                results['Vc'] = PropertyResolutionResult(
                    value=critical['Vc_cm3_mol'],
                    source='local',
                    method='acs_jced_5b00571_table1',
                    quality=0.99,
                    notes='ACS JCED 2015/IUPAC critical-property review Table 1 recommended value; units cm^3/mol',
                )
            if critical.get('Zc') is not None:
                results['Zc'] = PropertyResolutionResult(
                    value=critical['Zc'],
                    source='local',
                    method='acs_jced_5b00571_table1',
                    quality=0.99,
                    notes='ACS JCED 2015/IUPAC critical-property review Table 1 recommended value',
                )
            return results or None


        def _get_psrk_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, PropertyResolutionResult]]:
            cas = None
            for candidate in self._identifier_candidates(symbol, props):
                candidate_text = str(candidate).strip()
                if self._looks_like_cas(candidate_text):
                    cas = candidate_text
                    break
            values = psrk_unstarred_critical_values(cas)
            if not values:
                return None
            notes = (
                'Unstarred physical critical constant from the PSRK 2005 '
                'supplement; lower-priority fallback behind modern critical reviews'
            )
            return {
                prop_name: PropertyResolutionResult(
                    value=value,
                    source='local',
                    method='psrk2005_unstarred_critical',
                    quality=0.95,
                    notes=(
                        notes
                        + ('; units K' if prop_name == 'Tc' else '; units bar')
                    ),
                )
                for prop_name, value in values.items()
            }


        def _load_acs_jced_critical_table(self) -> Optional[Dict[str, Dict[str, Any]]]:
            cache = getattr(self, '_acs_jced_critical_table_cache', None)
            if cache is not None:
                return cache
            path = ACS_JCED_5B00571_TABLE1_PATH
            if not path.exists():
                self._acs_jced_critical_table_cache = None
                return None
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self._acs_jced_critical_table_cache = None
                return None
            chemicals = payload.get('chemicals') or {}
            self._acs_jced_critical_table_cache = chemicals
            return chemicals


        @staticmethod
        def _looks_like_cas(identifier: Any) -> bool:
            return bool(re.fullmatch(r'\d{2,7}-\d{2}-\d', str(identifier).strip()))


        def _effective_critical_candidates(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> list[tuple[str, str]]:
            candidates = self._identifier_candidates(symbol, props)
            result = []
            seen = set()

            def add(kind: str, value: Any) -> None:
                if not value:
                    return
                key = (kind, str(value).strip())
                if key not in seen:
                    seen.add(key)
                    result.append(key)

            for key in ('CAS', 'cas'):
                add('cas', props.get(key))
            for candidate in candidates:
                if self._looks_like_cas(candidate):
                    add('cas', candidate)

            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import get_compound_identity_resolver
                else:
                    from compound_identity import get_compound_identity_resolver
                identity_resolver = get_compound_identity_resolver()
                for candidate in candidates:
                    cas = identity_resolver.resolve_cas(str(candidate), allow_formula=False)
                    add('cas', cas)
            except Exception:
                pass

            for candidate in candidates:
                add('name', candidate)
            return result


        def _get_effective_critical_properties(
            self,
            symbol: str,
            props: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            path = EFFECTIVE_CRITICALS_DB_PATH
            if not path.exists():
                return None

            cache = getattr(self, '_effective_critical_cache', None)
            if cache is None:
                cache = {}
                self._effective_critical_cache = cache

            candidates = tuple(self._effective_critical_candidates(symbol, props))
            cache_key = (str(path), candidates)
            if cache_key in cache:
                cached = cache[cache_key]
                return dict(cached) if cached else None

            columns = (
                'CAS, name, MW, Tc, Pc, Vc, Zc, omega, '
                'Tc_quality, Pc_quality, Vc_quality, Zc_quality, omega_quality'
            )
            try:
                with closing(sqlite3.connect(path)) as connection:
                    connection.row_factory = sqlite3.Row
                    for kind, value in candidates:
                        if kind != 'cas':
                            continue
                        row = connection.execute(
                            f'SELECT {columns} FROM effective_criticals WHERE CAS = ?',
                            (value,),
                        ).fetchone()
                        if row:
                            result = dict(row)
                            cache[cache_key] = result
                            return dict(result)

                    for kind, value in candidates:
                        if kind != 'name':
                            continue
                        rows = connection.execute(
                            f'SELECT {columns} FROM effective_criticals WHERE lower(name) = lower(?)',
                            (value,),
                        ).fetchall()
                        if len(rows) == 1:
                            result = dict(rows[0])
                            cache[cache_key] = result
                            return dict(result)
            except sqlite3.Error:
                pass

            cache[cache_key] = None
            return None


        @staticmethod
        def _effective_critical_quality_note(quality: float) -> str:
            if quality >= 0.94:
                return 'physical-critical-equivalent quality score'
            if quality >= 0.90:
                return 'high-quality EOS-effective regression score'
            if quality >= 0.85:
                return 'EOS-effective parameter for difficult/nonphysical critical behavior'
            if quality >= 0.80:
                return 'lower-confidence EOS-effective parameter'
            return 'weak EOS-effective fallback parameter'


        def _effective_critical_result(
            self,
            row: Dict[str, Any],
            prop_name: str,
        ) -> PropertyResolutionResult:
            quality = self._clamp_quality(row.get(f'{prop_name}_quality'), 0.0)
            return PropertyResolutionResult(
                value=row[prop_name],
                source='local',
                method='effective_critical',
                quality=quality,
                notes=(
                    f"EOS-effective critical parameter from effective_criticals.sqlite "
                    f"for {row.get('name')} ({row.get('CAS')}); "
                    f"{self._effective_critical_quality_note(quality)}"
                ),
            )


        def _critical_input_result(
            self,
            props: Dict[str, Any],
            results: Dict[str, PropertyResolutionResult],
            prop_name: str,
        ) -> Optional[PropertyResolutionResult]:
            result = results.get(prop_name)
            if result and result.value is not None:
                return result
            return self._source_result_for_value(
                props,
                prop_name,
                units={'Tc': 'K', 'Pc': 'bar', 'Vc': 'cm^3/mol', 'Tb': 'K'}.get(prop_name, ''),
            )


        def _critical_tb_input_result(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            result = self._source_result_for_value(props, 'Tb', units='K')
            if result and result.value is not None:
                return result
            return self._estimate_boiling_point_fallback(props)


        @staticmethod
        def _formula_reports_net_charge(formula: Any) -> bool:
            if not formula:
                return False
            return bool(re.search(r'(?:[+-]\d*|\(\d*[+-]\))$', str(formula).strip()))


        @classmethod
        def _critical_identity_formula(cls, props: Dict[str, Any]) -> str:
            formula = str((props or {}).get('formula') or '').strip()
            if formula:
                return formula
            try:
                from chemicals.identifiers import search_chemical
            except ImportError:
                return ''
            for identifier in (
                (props or {}).get('CAS'),
                (props or {}).get('cas'),
                (props or {}).get('name'),
            ):
                if not identifier:
                    continue
                try:
                    chemical = search_chemical(str(identifier))
                except Exception:
                    continue
                formula = str(getattr(chemical, 'formula', '') or '').strip()
                if formula:
                    return formula
            return ''


        @classmethod
        def _critical_formula_is_organic(cls, props: Dict[str, Any]) -> bool:
            classification = classify_strict_molecular_organic(
                cas=(props or {}).get('CAS') or (props or {}).get('cas'),
                formula=cls._critical_identity_formula(props),
                smiles=(props or {}).get('smiles'),
            )
            return classification.is_organic


        @classmethod
        def _critical_formula_is_metal_or_salt(cls, props: Dict[str, Any]) -> bool:
            # Organometallic molecular compounds remain molecular fluids for
            # this admission policy.  A metal atom alone is not sufficient
            # when the structure/formula also proves C-H or C-halogen bonds.
            if cls._critical_formula_is_organic(props):
                return False
            counts = cls._parse_formula_counts_for_fallback(
                cls._critical_identity_formula(props)
            )
            if not counts:
                return False
            metal_or_metalloid = {
                'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',
                'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                'No', 'Lr',
            }
            if any(element in metal_or_metalloid for element in counts):
                return True
            return bool(
                len(counts) > 2
                and counts.get('N') == 1
                and counts.get('H') == 4
                and counts.get('C', 0) == 0
            )


        @classmethod
        def _critical_pressure_is_admissible(
            cls,
            props: Dict[str, Any],
            value: Any,
        ) -> bool:
            try:
                pressure_bar = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
                return False
            maximum = (
                ORGANIC_CRITICAL_PRESSURE_MAXIMUM_BAR
                if cls._critical_formula_is_organic(props)
                else INORGANIC_CRITICAL_PRESSURE_MAXIMUM_BAR
            )
            return pressure_bar <= maximum


        @staticmethod
        def _critical_compressibility_is_admissible(value: Any) -> bool:
            try:
                compressibility = float(value)
            except (TypeError, ValueError):
                return False
            return (
                math.isfinite(compressibility)
                and CRITICAL_COMPRESSIBILITY_MINIMUM
                <= compressibility
                <= CRITICAL_COMPRESSIBILITY_MAXIMUM
            )


        def _estimate_boiling_point_fallback(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            # Ions are banned from estimation outright: every empirical
            # rung below (including the formula-blind MW power law) is
            # fitted on neutral molecular species.
            if self._formula_reports_net_charge(props.get('formula') or props.get('Formula')):
                return None
            nannoolal_result = self._estimate_nannoolal_boiling_point(props)
            if nannoolal_result:
                return nannoolal_result
            formula_result = self._estimate_boiling_point_from_formula(props)
            if formula_result:
                return formula_result
            return self._estimate_boiling_point_from_mw(props)


        def _estimate_nannoolal_boiling_point(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            nannoolal = self._run_nannoolal_estimate(props, allow_online=False)
            if not nannoolal or nannoolal.get('tb_K') is None:
                return None
            smiles_result = nannoolal['smiles_result']
            return PropertyResolutionResult(
                value=nannoolal['tb_K'],
                source='estimated',
                method='nannoolal_tb',
                quality=self._nannoolal_quality([smiles_result], 0.80),
                notes=self._nannoolal_notes(
                    'Estimated by Nannoolal group contribution from SMILES',
                    nannoolal,
                ),
            )


        def _estimate_nannoolal_critical_property(
            self,
            props: Dict[str, Any],
            results: Dict[str, PropertyResolutionResult],
            prop_name: str,
            allow_online: bool = True,
        ) -> Optional[PropertyResolutionResult]:
            tb_result = None
            tb_value = None
            real_tb = False
            if prop_name == 'Tc':
                tb_result = self._source_result_for_value(props, 'Tb', units='K')
                if tb_result and tb_result.value is not None:
                    real_tb = self._result_is_real(tb_result)
                    if real_tb:
                        tb_value = tb_result.value
            nannoolal = self._run_nannoolal_estimate(props, tb=tb_value, allow_online=allow_online)
            if not nannoolal:
                return None
            smiles_result = nannoolal['smiles_result']
            if prop_name == 'Tc':
                value = nannoolal.get('tc_K')
                if value is None:
                    return None
                if real_tb:
                    quality = self._nannoolal_quality([smiles_result, tb_result], 0.85)
                    notes = 'Estimated by Nannoolal group contribution using real/source-backed Tb'
                else:
                    quality = self._nannoolal_quality([smiles_result], 0.75)
                    notes = 'Estimated by Nannoolal group contribution using internally estimated Tb'
                return PropertyResolutionResult(
                    value=value,
                    source='estimated',
                    method='nannoolal_tc',
                    quality=quality,
                    notes=self._nannoolal_notes(notes, nannoolal),
                )
            if prop_name == 'Pc':
                value = nannoolal.get('pc_kPa')
                if value is None:
                    return None
                return PropertyResolutionResult(
                    value=value / 100.0,
                    source='estimated',
                    method='nannoolal_pc',
                    quality=self._nannoolal_quality([smiles_result], 0.80),
                    notes=self._nannoolal_notes(
                        'Estimated by Nannoolal group contribution; converted from kPa to bar',
                        nannoolal,
                    ),
                )
            if prop_name == 'Vc':
                value = nannoolal.get('vc_cm3_mol')
                if value is None:
                    return None
                return PropertyResolutionResult(
                    value=value,
                    source='estimated',
                    method='nannoolal_vc',
                    quality=self._nannoolal_quality([smiles_result], 0.85),
                    notes=self._nannoolal_notes(
                        'Estimated by Nannoolal group contribution from SMILES',
                        nannoolal,
                    ),
                )
            return None


        def _nannoolal_quality(
            self,
            inputs: list[Optional[PropertyResolutionResult]],
            maximum: float,
        ) -> float:
            usable = [item for item in inputs if item is not None and item.value is not None]
            if not usable:
                return maximum
            return min(min(self._result_quality(item, maximum) for item in usable), maximum)


        @staticmethod
        def _nannoolal_notes(prefix: str, nannoolal: Dict[str, Any]) -> str:
            notes = [prefix, f"SMILES source: {nannoolal['smiles_result'].method}"]
            warnings = nannoolal.get('warnings') or []
            if warnings:
                notes.append('warnings: ' + '; '.join(str(item) for item in warnings))
            return '; '.join(notes)


        def _run_nannoolal_estimate(
            self,
            props: Dict[str, Any],
            tb: Optional[float] = None,
            allow_online: bool = True,
        ) -> Optional[Dict[str, Any]]:
            smiles_result = self._smiles_result_for_boiling_point(props)
            if not smiles_result:
                smiles_result = self._metadata_smiles_result_for_nannoolal(
                    props,
                    allow_online=allow_online,
                )
            if not smiles_result or not smiles_result.value:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..nannoolal_method import estimate as nannoolal_estimate, NannoolalError
                else:
                    from nannoolal_method import estimate as nannoolal_estimate, NannoolalError
                result = nannoolal_estimate(str(smiles_result.value), tb=float(tb) if tb is not None else None)
            except (ImportError, TypeError, ValueError, NannoolalError):
                return None
            return {
                'smiles_result': smiles_result,
                'tb_K': result.tb_K,
                'tc_K': result.tc_K,
                'pc_kPa': result.pc_kPa,
                'vc_cm3_mol': result.vc_cm3_mol,
                'warnings': result.warnings,
            }


        def _metadata_smiles_result_for_nannoolal(
            self,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> Optional[PropertyResolutionResult]:
            identifier = next(iter(self._hbd_lookup_identifiers(props)), '')
            return self._resolve_smiles_result(str(identifier), props, allow_online=allow_online)


        def _estimate_boiling_point_from_mw(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            """Very weak fallback when formula features are unavailable."""
            mw_result = self._source_result_for_value(props, 'MW', units='g/mol')
            if not mw_result or mw_result.value is None:
                return None
            try:
                mw = float(mw_result.value)
            except (TypeError, ValueError):
                return None
            if mw <= 0.0:
                return None
            value = TB_MW_FALLBACK_COEFFICIENT * mw ** TB_MW_FALLBACK_EXPONENT
            return PropertyResolutionResult(
                value=value,
                source='estimated',
                method='mw_correlation',
                quality=self._combine_quality([mw_result], method_factor=0.30),
                notes='weak final fallback from Perry-refit molecular-weight power law',
            )


        def _estimate_boiling_point_from_formula(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            formula_result = self._source_result_for_value(props, 'formula')
            formula = formula_result.value if formula_result else (props.get('Formula') or props.get('symbol'))
            counts = self._parse_formula_counts_for_fallback(formula)
            if not counts:
                return None
            heavy_atoms = sum(count for element, count in counts.items() if element != 'H')
            if heavy_atoms < 3:
                return None
            hbd_result = self._hbd_count_for_boiling_point(props, counts, allow_online=False)
            has_hbd = hbd_result is not None
            hbd_count = hbd_result[0] if hbd_result else 0
            coefficients = TB_FORMULA_HBD_COEFFICIENTS if has_hbd else TB_FORMULA_NO_HBD_COEFFICIENTS
            carbon_count = counts.get('C', 0)
            x_weighted = sum(counts.get(element, 0) * weight for element, weight in TB_HALOGEN_WEIGHTS.items())
            other_count = sum(
                count for element, count in counts.items()
                if element not in {'H', 'C', 'N', 'O', *TB_HALOGEN_WEIGHTS}
            )
            try:
                value = (
                    coefficients['A']
                    + coefficients['H'] * counts.get('H', 0)
                    + coefficients['C'] * carbon_count ** coefficients['C_exp']
                    + coefficients['N'] * counts.get('N', 0)
                    + coefficients['O'] * counts.get('O', 0)
                    + coefficients['X'] * x_weighted
                    + coefficients['other'] * other_count
                    + coefficients['HBD'] * hbd_count
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None

            inputs = [formula_result] if formula_result else []
            if has_hbd and hbd_result[1] is not None:
                inputs.append(hbd_result[1])
            method_factor = 0.55 if has_hbd else 0.45
            method = 'formula_hbd_boiling_point' if has_hbd else 'formula_no_hbd_boiling_point'
            hbd_note = (
                f"; HBD={hbd_count:g} from {hbd_result[2]}"
                if has_hbd else
                '; HBD unavailable, used no-HBD N/O refit'
            )
            return PropertyResolutionResult(
                value=value,
                source='estimated',
                method=method,
                quality=self._combine_quality(inputs, method_factor=method_factor),
                notes=(
                    f"Estimated from formula atom counts for {formula}; "
                    f"heavy atoms={heavy_atoms:g}{hbd_note}"
                ),
            )


        @staticmethod
        def _parse_formula_counts_for_fallback(formula: Any) -> Optional[Dict[str, int]]:
            if not formula:
                return None
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import parse_formula_counts
                else:
                    from compound_identity import parse_formula_counts
            except Exception:
                return None
            # parse_formula_counts refuses charge-suffixed formulas (e.g.
            # 'C2O4-2'); ions must stay unparsed here so that no empirical
            # fallback ever estimates properties for a charged species.
            return parse_formula_counts(str(formula)) or None


        def _hbd_count_for_boiling_point(
            self,
            props: Dict[str, Any],
            counts: Dict[str, int],
            allow_online: bool = True,
        ) -> Optional[tuple[int, Optional[PropertyResolutionResult], str]]:
            smiles_result = self._smiles_result_for_boiling_point(props)
            if smiles_result and smiles_result.value:
                hbd = self._rdkit_hbd_count(smiles_result.value)
                if hbd is not None:
                    return hbd, smiles_result, 'SMILES'

            fallback_props = dict(props)
            for key in (
                'smiles', 'SMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
                'canonical_smiles', 'CanonicalSMILES',
                'isomeric_smiles', 'IsomericSMILES',
            ):
                fallback_props.pop(key, None)
            identifier = next(iter(self._hbd_lookup_identifiers(fallback_props)), '')
            resolved = self._resolve_smiles_result(
                str(identifier),
                fallback_props,
                allow_online=allow_online,
            )
            if resolved and resolved.value:
                hbd = self._rdkit_hbd_count(resolved.value)
                if hbd is not None:
                    return hbd, resolved, resolved.method
            return None


        @staticmethod
        def _hbd_lookup_identifiers(props: Dict[str, Any]) -> list[Any]:
            identifiers = []
            for key in ('CAS', 'cas', 'name', 'symbol'):
                value = props.get(key)
                if key == 'symbol' and CriticalPropertiesMixin._hbd_identifier_looks_like_formula(value):
                    continue
                if value and value not in identifiers:
                    identifiers.append(value)
            for value in props.get('identifiers', []) or []:
                if CriticalPropertiesMixin._hbd_identifier_looks_like_formula(value):
                    continue
                if value and value not in identifiers:
                    identifiers.append(value)
            return identifiers


        @staticmethod
        def _hbd_identifier_looks_like_formula(identifier: Any) -> bool:
            if not identifier:
                return False
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..compound_identity import looks_like_formula
                else:
                    from compound_identity import looks_like_formula
                return looks_like_formula(str(identifier))
            except Exception:
                return CriticalPropertiesMixin._looks_like_molecular_formula(str(identifier))


        def _smiles_result_for_boiling_point(
            self,
            props: Dict[str, Any],
        ) -> Optional[PropertyResolutionResult]:
            for key in (
                'smiles', 'SMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
                'canonical_smiles', 'CanonicalSMILES',
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
            return None


        @staticmethod
        def _rdkit_hbd_count(smiles: Any) -> Optional[int]:
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return None
                return int(rdMolDescriptors.CalcNumHBD(mol))
            except Exception:
                return None


        def _lee_kesler_quality_factor(
            self,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> tuple[float, str]:
            smiles_result = self._smiles_result_for_boiling_point(props)
            if not smiles_result:
                smiles_result = self._metadata_smiles_result_for_nannoolal(
                    props,
                    allow_online=allow_online,
                )
            counts = self._rdkit_hbond_counts(
                smiles_result.value if smiles_result else None
            )
            if counts is None:
                return 0.80, 'Lee-Kesler quality factor=0.80; HBA/HBD unavailable'
            hba_count, hbd_count = counts
            factor = 0.80 if hbd_count > 1 or hba_count > 2 else 0.95
            return (
                factor,
                f'Lee-Kesler quality factor={factor:.2f}; '
                f'HBA={hba_count:g}, HBD={hbd_count:g}',
            )


        @staticmethod
        def _rdkit_hbond_counts(smiles: Any) -> Optional[tuple[int, int]]:
            if not smiles:
                return None
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return None
                return (
                    int(rdMolDescriptors.CalcNumHBA(mol)),
                    int(rdMolDescriptors.CalcNumHBD(mol)),
                )
            except Exception:
                return None


        def _critical_zc_identity_result(
            self,
            props: Dict[str, Any],
            results: Dict[str, PropertyResolutionResult],
        ) -> Optional[PropertyResolutionResult]:
            Tc_result = self._critical_input_result(props, results, 'Tc')
            Pc_result = self._critical_input_result(props, results, 'Pc')
            Vc_result = self._critical_input_result(props, results, 'Vc')
            if not (Tc_result and Pc_result and Vc_result):
                return None
            if Tc_result.value is None or Pc_result.value is None or Vc_result.value is None:
                return None
            try:
                Zc = (float(Pc_result.value) * 100000.0) * (float(Vc_result.value) * 1.0e-6) / (R * float(Tc_result.value))
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            if not self._critical_compressibility_is_admissible(Zc):
                return None
            inputs = [Tc_result, Pc_result, Vc_result]
            all_coolprop = all(
                str(item.method).startswith('coolprop_')
                for item in inputs
            )
            return PropertyResolutionResult(
                value=Zc,
                source=(
                    'calculated'
                    if all_coolprop
                    else self._derived_source(inputs, exact_formula=True)
                ),
                method=(
                    'coolprop_critical_identity'
                    if all_coolprop
                    else 'critical_volume_identity'
                ),
                quality=self._combine_quality(inputs, exact_formula=True),
                notes=(
                    'Zc = Pc*Vc/(R*Tc) from the selected CoolProp Tc/Pc/Vc bundle'
                    if all_coolprop
                    else 'Zc = Pc*Vc/(R*Tc) from the final selected Tc/Pc/Vc bundle'
                ),
            )


        @classmethod
        def _critical_source_is_estimated(cls, props: Dict[str, Any], prop_name: str) -> bool:
            sources = (props or {}).get('property_sources') or {}
            source = sources.get(prop_name) or {}
            return cls._source_meta_is_soft(source)


        def _critical_result_is_estimated(
            self,
            prop_name: str,
            result: Optional[PropertyResolutionResult],
            props: Dict[str, Any],
        ) -> bool:
            if not result or result.value is None:
                return True
            if self._result_is_soft(result):
                return True
            return self._critical_source_is_estimated(props, prop_name)


        @classmethod
        def _critical_result_can_replace(
            cls,
            existing: Optional[PropertyResolutionResult],
            candidate: Optional[PropertyResolutionResult],
        ) -> bool:
            if candidate is None or candidate.value is None:
                return False
            if existing is None or existing.value is None:
                return True
            if not cls._result_is_soft(existing):
                return False
            return cls._result_quality(candidate, 0.0) > cls._result_quality(existing, 0.0) + 1e-12


        @classmethod
        def _pc_fallback_non_hydrogen_atoms(cls, formula: str) -> Optional[int]:
            counts = cls._parse_formula_counts_for_fallback(formula)
            if not counts:
                return None
            total = sum(count for element, count in counts.items() if element != 'H')
            return total if total >= 3 else None


        @staticmethod
        def _pc_fallback_smiles_ring_score(smiles: Optional[str]) -> Optional[int]:
            if not smiles:
                return None
            try:
                from rdkit import Chem
            except Exception:
                return None
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            score = 0
            for ring in mol.GetRingInfo().AtomRings():
                if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
                    score += 2
                else:
                    score += 1
            return score


        @staticmethod
        def _pc_fallback_normalized_name(name: str) -> str:
            text = str(name or '').lower()
            text = re.sub(r'[^a-z0-9]+', ' ', text)
            return re.sub(r'\s+', ' ', text).strip()


        @classmethod
        def _pc_fallback_name_ring_score(cls, name: str) -> int:
            text = cls._pc_fallback_normalized_name(name)
            token_groups = (
                (5, ('pentacyclo',)),
                (4, ('tetracyclo', 'diphenyl', 'biphen', 'naph')),
                (3, ('tricyclo',)),
                (6, ('terphen', 'anth')),
                (
                    2,
                    (
                        'bicyclo', 'resorcinol', 'catechol', 'thiophene',
                        'styrene', 'quinone', 'toluene', 'cumene', 'cresol',
                        'xylene', 'pyrrole', 'phthal', 'furan', 'pyrid',
                        'benz', 'phen', 'ani',
                    ),
                ),
                (1, ('pyrrolidine', 'piperidine', 'ene oxide', 'dioxane', 'cyclo')),
            )
            tokens = sorted(
                ((token, value) for value, group in token_groups for token in group),
                key=lambda item: (-len(item[0]), -item[1]),
            )

            def tetrahydro_before(start: int) -> bool:
                prefix = text[max(0, start - 35):start]
                return bool(
                    re.search(r'tetrahydro\s*$', prefix)
                    or re.search(r'tetrahydro\s+[a-z0-9 ]*$', prefix)
                )

            candidates = []
            for token, value in tokens:
                pattern = re.escape(token).replace(r'\ ', r'\s+')
                for match in re.finditer(pattern, text):
                    if token == 'phen' and 'phenanth' in text and match.start() == 0:
                        continue
                    if token == 'phthal' and 'naphthal' in text[max(0, match.start() - 3):match.end() + 3]:
                        continue
                    adjusted = value - 1 if value >= 2 and tetrahydro_before(match.start()) else value
                    candidates.append((match.start(), match.end(), adjusted))

            candidates.sort(key=lambda item: (-(item[1] - item[0]), -item[2], item[0]))
            used_spans = []
            score = 0
            for start, end, value in candidates:
                if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_spans):
                    continue
                used_spans.append((start, end))
                score += value
            return score


        def _estimate_pc_from_formula_tb(
            self,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> Optional[tuple[float, str]]:
            tb_result = self._critical_tb_input_result(props)
            if not tb_result or tb_result.value is None:
                return None
            atom_count = self._pc_fallback_non_hydrogen_atoms(props.get('formula') or props.get('symbol'))
            if atom_count is None:
                return None
            smiles_result = self._resolve_smiles_result(
                str(props.get('name') or props.get('symbol') or ''),
                props,
                allow_online=allow_online,
            )
            ring_score = self._pc_fallback_smiles_ring_score(
                smiles_result.value if smiles_result else None
            )
            ring_source = 'SMILES' if ring_score is not None else 'name'
            if ring_score is None:
                ring_score = self._pc_fallback_name_ring_score(props.get('name') or props.get('symbol'))

            try:
                Tb = float(tb_result.value)
                reduced_tb = Tb / PC_FALLBACK_TREF_K
                ring_factor = PC_FALLBACK_RING_A + PC_FALLBACK_RING_B * (reduced_tb - 1.0)
                value = (
                    205.0 / atom_count
                    * ring_factor ** ring_score
                    * reduced_tb ** PC_FALLBACK_TB_EXPONENT
                )
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None
            if not math.isfinite(value) or value <= 0.0 or ring_factor <= 0.0:
                return None
            notes = f'Estimated from formula atom count, Tb, and {ring_source}-derived ring score'
            return value, notes


        def _estimate_vc_from_formula_structure(
            self,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> Optional[tuple[float, list[PropertyResolutionResult], bool, str]]:
            formula_result = self._source_result_for_value(props, 'formula')
            formula = formula_result.value if formula_result else (props.get('Formula') or props.get('symbol'))
            counts = self._parse_formula_counts_for_fallback(formula)
            if not counts:
                return None
            heavy_atoms = sum(count for element, count in counts.items() if element != 'H')
            if heavy_atoms < 3:
                return None

            large_ring_count, ring_result, ring_note = self._large_ring_count_for_vc(
                props,
                allow_online=allow_online,
            )
            coeffs = VC_FORMULA_FALLBACK_COEFFICIENTS
            s_cl_p_count = counts.get('S', 0) + counts.get('Cl', 0) + counts.get('P', 0)
            other_count = sum(
                count for element, count in counts.items()
                if element not in {'H', 'C', 'O', 'N', 'F', 'Si', 'S', 'Cl', 'P'}
            )
            try:
                value = (
                    coeffs['A']
                    + coeffs['H'] * counts.get('H', 0)
                    + coeffs['C'] * counts.get('C', 0)
                    + coeffs['O'] * counts.get('O', 0)
                    + coeffs['N'] * counts.get('N', 0)
                    + coeffs['F'] * counts.get('F', 0)
                    + coeffs['Si'] * counts.get('Si', 0)
                    + coeffs['SClP'] * s_cl_p_count
                    + coeffs['other'] * other_count
                    + coeffs['large_ring'] * large_ring_count
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None

            inputs = [formula_result] if formula_result else []
            ring_available = ring_result is not None
            if ring_result:
                inputs.append(ring_result)
            notes = (
                f"Estimated from formula atom counts for {formula}; "
                f"heavy atoms={heavy_atoms:g}; large rings={large_ring_count:g} ({ring_note})"
            )
            return value, inputs, ring_available, notes


        def _large_ring_count_for_vc(
            self,
            props: Dict[str, Any],
            allow_online: bool = True,
        ) -> tuple[int, Optional[PropertyResolutionResult], str]:
            smiles_result = self._smiles_result_for_boiling_point(props)
            if smiles_result and smiles_result.value:
                count = self._rdkit_large_ring_count(smiles_result.value)
                if count is not None:
                    return count, smiles_result, 'SMILES'

            fallback_props = dict(props)
            for key in (
                'smiles', 'SMILES',
                'connectivity_smiles', 'ConnectivitySMILES',
                'canonical_smiles', 'CanonicalSMILES',
                'isomeric_smiles', 'IsomericSMILES',
            ):
                fallback_props.pop(key, None)
            identifier = next(iter(self._hbd_lookup_identifiers(fallback_props)), '')
            resolved = self._resolve_smiles_result(
                str(identifier),
                fallback_props,
                allow_online=allow_online,
            )
            if resolved and resolved.value:
                count = self._rdkit_large_ring_count(resolved.value)
                if count is not None:
                    return count, resolved, resolved.method
            return 0, None, 'no parseable SMILES; assumed acyclic for fallback'


        @staticmethod
        def _rdkit_large_ring_count(smiles: Any) -> Optional[int]:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return None
                return sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) >= 5)
            except Exception:
                return None


        def _fetch_critical_pubchem(self, symbol: str) -> Optional[Dict[str, float]]:
            """Compatibility wrapper around the shared strict online parser."""
            online = self._fetch_phase_change_pubchem(symbol)
            if not online:
                return None
            return {
                key: online[key]
                for key in ('Tc', 'Pc', 'Vc')
                if online.get(key) is not None
            } or None


        def _extract_critical_from_section(self, section: Dict, result: Dict):
            """Extract critical properties through the shared strict parser."""
            parsed = {}
            self._extract_phase_change_from_pubchem_node(section, parsed)
            self._finalize_pubchem_critical_candidates(parsed)
            for key in ('Tc', 'Pc', 'Vc'):
                if parsed.get(key) is not None:
                    result[key] = parsed[key]


        def _extract_pubchem_value(self, section: Dict) -> Optional[float]:
            """Extract numeric value from PubChem section"""
            for info in section.get('Information', []):
                value = info.get('Value', {})

                # Try Number format
                if 'Number' in value:
                    nums = value['Number']
                    if nums:
                        return float(nums[0])

                # Try StringWithMarkup format
                if 'StringWithMarkup' in value:
                    text = value['StringWithMarkup'][0].get('String', '')
                    match = re.search(r'([\d.]+)', text)
                    if match:
                        return float(match.group(1))

            return None


        def _extract_pubchem_temperature(self, section: Dict) -> Optional[float]:
            """Extract a temperature from a PubChem section and return Kelvin."""
            for text in self._pubchem_section_texts(section):
                value = self._parse_temperature_K(text)
                if value is not None:
                    return value
            return self._extract_pubchem_value(section)


        def _extract_pubchem_pressure(self, section: Dict) -> Optional[float]:
            """Extract a pressure from a PubChem section and return bar."""
            for text in self._pubchem_section_texts(section):
                value = self._parse_pressure_bar(text)
                if value is not None:
                    return value
            return self._extract_pubchem_value(section)


        @staticmethod
        def _pubchem_section_texts(section: Dict) -> list[str]:
            texts = []
            for info in section.get('Information', []):
                value = info.get('Value', {})
                unit = value.get('Unit', '')
                for marked in value.get('StringWithMarkup', []):
                    text = marked.get('String')
                    if text:
                        texts.append(text)
                for number in value.get('Number', []):
                    texts.append(f"{number} {unit}".strip())
            return texts


        @staticmethod
        def _parse_temperature_K(text: str) -> Optional[float]:
            text = text.replace('−', '-').replace(',', '')
            match = re.search(
                r'(-?\d+(?:\.\d+)?)\s*'
                r'(?:(?:°|deg(?:ree)?s?)\s*)?'
                r'(K|Kelvin|C|Celsius|F|Fahrenheit)\b',
                text,
                re.IGNORECASE,
            )
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit in {'k', 'kelvin'}:
                return value
            if unit in {'c', 'celsius'}:
                return value + 273.15
            if unit in {'f', 'fahrenheit'}:
                return (value - 32.0) * 5.0 / 9.0 + 273.15
            return None


        @staticmethod
        def _parse_pressure_bar(text: str) -> Optional[float]:
            text = text.replace('−', '-').replace(',', '').replace('X10+', 'e')
            match = re.search(
                r'(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(MPa|kPa|Pa|bar|atm|mmHg|psi)\b',
                text,
                re.IGNORECASE,
            )
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == 'bar':
                return value
            if unit == 'mpa':
                return value * 10.0
            if unit == 'kpa':
                return value / 100.0
            if unit == 'pa':
                return value / 100000.0
            if unit == 'atm':
                return value * NORMAL_BOILING_PRESSURE_BAR
            if unit == 'mmhg':
                return value / 750.062
            if unit == 'psi':
                return value * 0.0689476
            return None


        @staticmethod
        def _parse_volume_cm3_per_mol(text: str) -> Optional[float]:
            text = text.replace('−', '-').replace(',', '').replace('X10+', 'e')
            match = re.search(
                r'(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*'
                r'(cm3/mol|cm\^3/mol|cm³/mol|mL/mol|ml/mol|L/mol|l/mol|m3/kmol|m\^3/kmol)\b',
                text,
                re.IGNORECASE,
            )
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit in {'cm3/mol', 'cm^3/mol', 'cm³/mol', 'ml/mol'}:
                return value
            if unit in {'l/mol', 'L/mol'.lower()}:
                return value * 1000.0
            if unit in {'m3/kmol', 'm^3/kmol'}:
                return value * 1000.0
            return None


        @staticmethod
        def _parse_energy_kj_per_mol(text: str) -> Optional[float]:
            text = text.replace('−', '-').replace(',', '').replace('X10+', 'e')
            match = re.search(
                r'(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*'
                r'(kJ/mol|J/mol|J/kmol|kcal/mol|cal/mol|gcal/gmole)\b',
                text,
                re.IGNORECASE,
            )
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == 'kj/mol':
                return value
            if unit == 'j/mol':
                return value / 1000.0
            if unit == 'j/kmol':
                return value / 1_000_000.0
            if unit == 'kcal/mol':
                return value * 4.184
            if unit in {'cal/mol', 'gcal/gmole'}:
                return value * 4.184 / 1000.0
            return None


        def _estimate_omega(self, Tc: float, Pc: float, Tb: float) -> Optional[float]:
            """
            Estimate acentric factor from critical properties and boiling point.

            Uses Lee-Kesler correlation:
            omega = (ln(Pc/Psat(Tb)) - f0(Tbr)) / f1(Tbr), with Psat(Tb) = 1 atm

            where Tbr = Tb/Tc
            """
            try:
                Tbr = Tb / Tc

                # Don't extrapolate too far
                if Tbr < 0.5 or Tbr > 0.95:
                    return None

                f0 = 5.92714 - 6.09648/Tbr - 1.28862*math.log(Tbr) + 0.169347*(Tbr**6)
                f1 = 15.2518 - 15.6875/Tbr - 13.4721*math.log(Tbr) + 0.43577*(Tbr**6)

                if abs(f1) < 1e-10:
                    return None

                # At Tb, P_sat = 1 atm, so Pr = P_sat / Pc.
                omega = (math.log(NORMAL_BOILING_PRESSURE_BAR/Pc) - f0) / f1

                # Sanity check
                if omega < -0.5 or omega > 1.5:
                    return None

                return omega
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None
