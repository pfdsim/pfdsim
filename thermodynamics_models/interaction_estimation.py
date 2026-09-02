"""Frozen UNIFAC-to-molecular-activity interaction surrogates."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares


_MODIFIED_SOURCES = frozenset({'UNIFDMD', 'UNIFM2', 'UNIFNIST'})
_SOURCE_CLASSES = None
_FIT_CACHE_SCHEMA_VERSION = 2
_FIT_CACHE = None
_FIT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / 'data'
    / 'runtime'
    / 'interaction_estimation_cache.sqlite'
)
_SOURCE_PARAMETER_FINGERPRINTS: dict[str, str] = {}
_COMPOSITION_GRID = (
    1.0e-4, 1.0e-3, 1.0e-2, 0.05, 0.15, 0.30, 0.50,
    0.70, 0.85, 0.95, 0.99, 0.999, 0.9999,
)


def _source_classes():
    global _SOURCE_CLASSES
    if _SOURCE_CLASSES is None:
        from .unifac_models import (
            UNIFACThermodynamics,
            UNIFAC2Thermodynamics,
            UNIFDMDThermodynamics,
            UNIFM2Thermodynamics,
            UNIFNISTThermodynamics,
        )
        _SOURCE_CLASSES = {
            'UNIFAC': UNIFACThermodynamics,
            'UNIFAC2': UNIFAC2Thermodynamics,
            'UNIFDMD': UNIFDMDThermodynamics,
            'UNIFM2': UNIFM2Thermodynamics,
            'UNIFNIST': UNIFNISTThermodynamics,
        }
    return _SOURCE_CLASSES


def _source_component_groups(
    destination_thermo,
    source_name: str,
    source_class,
    component: str,
    unifac_groups: Optional[dict],
) -> tuple[Optional[dict], bool]:
    """Resolve one source component without constructing the source model."""
    if unifac_groups and component in unifac_groups:
        return dict(unifac_groups[component]), False
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .unifac_models import (
            can_exclude_component_from_unifac,
            resolve_component_unifac_groups,
        )
    else:
        from thermodynamics_models.unifac_models import (
            can_exclude_component_from_unifac,
            resolve_component_unifac_groups,
        )
    props = destination_thermo.props.get(component)
    groups = resolve_component_unifac_groups(
        component,
        props,
        destination_thermo.db,
        source_class.unifac_variant,
    )
    excluded = groups is None and can_exclude_component_from_unifac(props)
    return groups, excluded


def _fit_cache():
    global _FIT_CACHE
    if _FIT_CACHE is None or Path(_FIT_CACHE.path) != Path(_FIT_CACHE_PATH):
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..property_resolution.runtime_cache import SQLiteJSONCache
        else:
            from property_resolution.runtime_cache import SQLiteJSONCache
        _FIT_CACHE = SQLiteJSONCache(
            _FIT_CACHE_PATH,
            f'interaction_estimation_v{_FIT_CACHE_SCHEMA_VERSION}',
        )
    return _FIT_CACHE


def _source_parameter_fingerprint(source_name: str, source_type) -> str:
    cached = _SOURCE_PARAMETER_FINGERPRINTS.get(source_name)
    if cached is not None:
        return cached
    filename = str(getattr(source_type, 'default_unifac_data_filename', '') or '')
    path = Path(__file__).resolve().parent.parent / 'data' / filename
    digest = hashlib.sha256()
    digest.update(source_name.encode('utf-8'))
    digest.update(filename.encode('utf-8'))
    if path.is_file():
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    fingerprint = digest.hexdigest()
    _SOURCE_PARAMETER_FINGERPRINTS[source_name] = fingerprint
    return fingerprint


def _normalized_groups(groups: dict) -> list[list[object]]:
    return sorted(
        ([str(group), int(count)] for group, count in groups.items()),
        key=lambda item: item[0],
    )


def _component_fit_identity(thermo, component: str) -> dict:
    props = thermo.props.get(component)
    return {
        'CAS': str(getattr(props, 'CAS', '') or ''),
        'smiles': str(getattr(props, 'smiles', '') or ''),
        'formula': str(getattr(props, 'formula', '') or ''),
        'name': str(getattr(props, 'name', '') or ''),
    }


def _fit_cache_key(
    destination_thermo,
    destination: str,
    comp1: str,
    comp2: str,
    options: dict,
    source_type,
    source_groups: dict[str, dict],
) -> str:
    source_name = str(options['source'])
    payload = {
        'schema_version': _FIT_CACHE_SCHEMA_VERSION,
        'destination': destination,
        'source': source_name,
        'source_parameter_fingerprint': _source_parameter_fingerprint(
            source_name,
            source_type,
        ),
        'components': [
            {
                'identity': _component_fit_identity(destination_thermo, component),
                'source_groups': _normalized_groups(source_groups[component]),
                'r': (
                    float(destination_thermo.r[component])
                    if destination == 'UNIQUAC' else None
                ),
                'q': (
                    float(destination_thermo.q[component])
                    if destination == 'UNIQUAC' else None
                ),
            }
            for component in (comp1, comp2)
        ],
        'Tmin_K': float(options['Tmin_K']),
        'Tmax_K': float(options['Tmax_K']),
        'parameter_order': str(options.get('parameter_order', 'source')),
        'alpha12': (
            float(options.get('alpha12', 0.3))
            if destination == 'NRTL' else None
        ),
        'T_ref_K': float(options.get('T_ref_K', 298.15)),
        'composition_grid': list(_COMPOSITION_GRID),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _temperature_grid(low: float, high: float, modified: bool) -> tuple[float, ...]:
    count = 7 if modified else 5
    if abs(high - low) <= 1.0e-12:
        return (float(low),)
    return tuple(float(value) for value in np.linspace(low, high, count))


def _uniquac_ln_gamma(
    x1: float,
    r: tuple[float, float],
    q: tuple[float, float],
    tau12: float,
    tau21: float,
) -> tuple[float, float]:
    x = (x1, 1.0 - x1)
    rx = sum(r[index] * x[index] for index in range(2)) or 1.0e-30
    qx = sum(q[index] * x[index] for index in range(2)) or 1.0e-30
    z = 10.0
    ell = tuple(
        0.5 * z * (r[index] - q[index]) - (r[index] - 1.0)
        for index in range(2)
    )
    xl = sum(x[index] * ell[index] for index in range(2))
    theta = tuple(q[index] * x[index] / qx for index in range(2))
    tau = ((1.0, tau12), (tau21, 1.0))
    theta_tau_col = tuple(
        sum(theta[j] * tau[j][i] for j in range(2)) or 1.0e-30
        for i in range(2)
    )
    result = []
    for i in range(2):
        phi_over_x = max(r[i] / rx, 1.0e-30)
        theta_over_phi = max(q[i] * rx / (r[i] * qx), 1.0e-30)
        combinatorial = (
            math.log(phi_over_x)
            + 0.5 * z * q[i] * math.log(theta_over_phi)
            + ell[i]
            - phi_over_x * xl
        )
        residual_sum = sum(
            theta[j] * tau[i][j] / theta_tau_col[j]
            for j in range(2)
        )
        residual = q[i] * (
            1.0 - math.log(theta_tau_col[i]) - residual_sum
        )
        result.append(combinatorial + residual)
    return float(result[0]), float(result[1])


def _nrtl_ln_gamma(
    x1: float,
    tau12: float,
    tau21: float,
    alpha: float,
) -> tuple[float, float]:
    x = (x1, 1.0 - x1)
    tau = ((0.0, tau12), (tau21, 0.0))
    G = (
        (1.0, math.exp(max(min(-alpha * tau12, 50.0), -50.0))),
        (math.exp(max(min(-alpha * tau21, 50.0), -50.0)), 1.0),
    )
    denominators = tuple(
        sum(x[k] * G[k][j] for k in range(2)) or 1.0e-30
        for j in range(2)
    )
    weighted_tau = tuple(
        sum(x[m] * tau[m][j] * G[m][j] for m in range(2))
        / denominators[j]
        for j in range(2)
    )
    result = []
    for i in range(2):
        value = sum(
            x[j] * tau[j][i] * G[j][i] / denominators[i]
            for j in range(2)
        )
        value += sum(
            x[j] * G[i][j] / denominators[j]
            * (tau[i][j] - weighted_tau[j])
            for j in range(2)
        )
        result.append(value)
    return float(result[0]), float(result[1])


def _physical_parameters(
    scaled: np.ndarray,
    destination: str,
    modified: bool,
) -> tuple[float, ...]:
    if not modified:
        return tuple(float(value) * 1000.0 for value in scaled)
    if destination == 'UNIQUAC':
        scales = (1.0, 1000.0, 1.0e-3, 1.0, 1000.0, 1.0e-3)
    else:
        scales = (1.0, 1000.0, 1.0e-3, 1.0, 1000.0, 1.0e-3)
    return tuple(float(value) * scales[index] for index, value in enumerate(scaled))


def _destination_ln_gamma(
    destination: str,
    modified: bool,
    physical: tuple[float, ...],
    T: float,
    x1: float,
    *,
    r: Optional[tuple[float, float]],
    q: Optional[tuple[float, float]],
    alpha: float,
    T_ref: float,
) -> tuple[float, float]:
    if not modified:
        a12, a21 = physical
        if destination == 'UNIQUAC':
            tau12 = math.exp(max(min(-a12 / (1.98720425864083 * T), 50.0), -50.0))
            tau21 = math.exp(max(min(-a21 / (1.98720425864083 * T), 50.0), -50.0))
            return _uniquac_ln_gamma(x1, r, q, tau12, tau21)
        return _nrtl_ln_gamma(
            x1,
            a12 / (1.98720425864083 * T),
            a21 / (1.98720425864083 * T),
            alpha,
        )

    p12 = physical[:3]
    p21 = physical[3:]
    if destination == 'UNIQUAC':
        exponent12 = p12[0] + p12[1] / T + p12[2] * T
        exponent21 = p21[0] + p21[1] / T + p21[2] * T
        return _uniquac_ln_gamma(
            x1,
            r,
            q,
            math.exp(max(min(exponent12, 50.0), -50.0)),
            math.exp(max(min(exponent21, 50.0), -50.0)),
        )
    tau12 = p12[0] + p12[1] / T + p12[2] * T
    tau21 = p21[0] + p21[1] / T + p21[2] * T
    return _nrtl_ln_gamma(x1, tau12, tau21, alpha)


def _fit_pair(
    destination_thermo,
    destination: str,
    comp1: str,
    comp2: str,
    options: dict,
    unifac_groups: Optional[dict],
    resolved_source_groups: Optional[dict[str, dict]] = None,
) -> tuple[dict, dict]:
    source_name = options['source']
    source_class = _source_classes()[source_name]
    source_groups = dict(resolved_source_groups or {})
    if resolved_source_groups is None:
        for component in (comp1, comp2):
            groups, excluded = _source_component_groups(
                destination_thermo,
                source_name,
                source_class,
                component,
                unifac_groups,
            )
            if excluded:
                raise ValueError(
                    f"{source_name} excludes {component} from its liquid group model "
                    "(for example as a contextual Henry solute); no molecular "
                    "liquid interaction can be regressed"
                )
            if groups is None:
                raise ValueError(
                    f"{source_name} groups not found for '{component}'"
                )
            source_groups[component] = groups
    cache_key = _fit_cache_key(
        destination_thermo,
        destination,
        comp1,
        comp2,
        options,
        source_class,
        source_groups,
    )
    try:
        cached = _fit_cache().get(cache_key)
    except Exception:
        cached = None
    if isinstance(cached, dict):
        if cached.get('_missing'):
            raise ValueError(str(cached.get('error') or 'cached fit failure'))
        record = cached.get('record')
        metadata = cached.get('metadata')
        if isinstance(record, dict) and isinstance(metadata, dict):
            record = dict(record)
            metadata = dict(metadata)
            record['component1'] = comp1
            record['component2'] = comp2
            metadata['component1'] = comp1
            metadata['component2'] = comp2
            metadata['fit_cache_hit'] = True
            metadata['fit_cache_key'] = cache_key
            return record, metadata
    source = source_class(
        [comp1, comp2],
        destination_thermo.db,
        source_groups,
    )
    modified = source_name in _MODIFIED_SOURCES
    temperatures = _temperature_grid(
        options['Tmin_K'], options['Tmax_K'], modified
    )
    targets = []
    for T in temperatures:
        for x1 in _COMPOSITION_GRID:
            gamma = source.activity_coefficients(
                T, {comp1: x1, comp2: 1.0 - x1}
            )
            targets.append((
                T,
                x1,
                math.log(max(float(gamma[comp1]), 1.0e-300)),
                math.log(max(float(gamma[comp2]), 1.0e-300)),
            ))

    r = q = None
    if destination == 'UNIQUAC':
        r = (destination_thermo.r[comp1], destination_thermo.r[comp2])
        q = (destination_thermo.q[comp1], destination_thermo.q[comp2])
    alpha = float(options.get('alpha12', 0.3))
    T_ref = float(options.get('T_ref_K', 298.15))
    count = 6 if modified else 2

    def residual(scaled):
        physical = _physical_parameters(scaled, destination, modified)
        values = []
        for T, x1, target1, target2 in targets:
            predicted1, predicted2 = _destination_ln_gamma(
                destination,
                modified,
                physical,
                T,
                x1,
                r=r,
                q=q,
                alpha=alpha,
                T_ref=T_ref,
            )
            values.extend((predicted1 - target1, predicted2 - target2))
        return np.asarray(values, dtype=float)

    result = least_squares(
        residual,
        np.zeros(count, dtype=float),
        bounds=(-20.0, 20.0),
        max_nfev=2000,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        error = f"least-squares fit failed: {result.message}"
        try:
            _fit_cache().set(
                cache_key,
                {
                    '_missing': True,
                    'error': error,
                    '_cache_family': 'interaction_estimation',
                    '_provider': source_name,
                    '_contract_version': _FIT_CACHE_SCHEMA_VERSION,
                    '_identifier_key': cache_key,
                },
            )
        except Exception:
            pass
        raise ValueError(error)
    physical = _physical_parameters(result.x, destination, modified)
    fit_residual = residual(result.x)
    rmse = float(math.sqrt(float(np.mean(fit_residual * fit_residual))))
    maximum = float(np.max(np.abs(fit_residual)))
    record = {
        'component1': comp1,
        'component2': comp2,
        'model': destination,
        'comment': f"Estimated from {source_name} binary activity coefficients",
    }
    if not modified:
        record.update(
            a12_cal_per_mol=physical[0],
            a21_cal_per_mol=physical[1],
        )
    elif destination == 'UNIQUAC':
        record.update(
            tau12_a=physical[0], tau12_b=physical[1], tau12_c=0.0,
            tau12_d=physical[2], tau12_e=0.0,
            tau21_a=physical[3], tau21_b=physical[4], tau21_c=0.0,
            tau21_d=physical[5], tau21_e=0.0, tau_tref=T_ref,
            model_variant='standard_uniquac', use_q_prime=False,
        )
    else:
        record.update(
            alpha12=alpha,
            tau12_c=physical[0], tau12_d=physical[1], tau12_e=0.0,
            tau12_f=physical[2], tau12_g=0.0,
            tau21_c=physical[3], tau21_d=physical[4], tau21_e=0.0,
            tau21_f=physical[5], tau21_g=0.0,
            tau_tref=T_ref,
        )
    if destination == 'NRTL' and not modified:
        record['alpha12'] = alpha
    metadata = {
        'component1': comp1,
        'component2': comp2,
        'model': destination,
        'source': source_name,
        'fit_Tmin_K': float(options['Tmin_K']),
        'fit_Tmax_K': float(options['Tmax_K']),
        'parameter_order': 'source',
        'alpha12': alpha if destination == 'NRTL' else None,
        'rmse_ln_gamma': rmse,
        'max_abs_ln_gamma_error': maximum,
        'sample_count': len(targets),
        'fitted_once': True,
        'fit_cache_hit': False,
        'fit_cache_key': cache_key,
    }
    try:
        _fit_cache().set(
            cache_key,
            {
                'record': record,
                'metadata': metadata,
                '_cache_family': 'interaction_estimation',
                '_provider': source_name,
                '_contract_version': _FIT_CACHE_SCHEMA_VERSION,
                '_identifier_key': cache_key,
            },
        )
    except Exception:
        pass
    return record, metadata


def estimate_missing_interactions(
    destination_thermo,
    destination: str,
    rules: Optional[list[dict]],
    existing_interaction: Callable[[str, str], Optional[dict]],
    unifac_groups: Optional[dict] = None,
    explicit_pair_keys: Optional[set[tuple[str, str]]] = None,
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    """Fit each missing binary once and return frozen interaction records."""
    if not rules:
        return [], {}
    global_rules = [
        dict(rule) for rule in rules
        if rule.get('model') == destination and 'component1' not in rule
    ]
    if not global_rules:
        return [], {}
    global_rule = global_rules[0]
    pair_rules = {
        tuple(sorted((rule['component1'], rule['component2']))): dict(rule)
        for rule in rules
        if rule.get('model') == destination and 'component1' in rule
    }
    records = []
    metadata = {}
    failures = []
    source_group_cache: dict[
        tuple[str, str], tuple[Optional[dict], bool]
    ] = {}
    reported_source_exclusions: set[tuple[str, str]] = set()
    for comp1, comp2 in combinations(destination_thermo.components, 2):
        options = dict(global_rule)
        options.update(pair_rules.get(tuple(sorted((comp1, comp2))), {}))
        pair_key = tuple(sorted((comp1, comp2)))
        if pair_key in (explicit_pair_keys or set()):
            continue
        policy = str(options.get('policy', 'missing_only'))
        if policy == 'missing_only' and existing_interaction(comp1, comp2) is not None:
            continue
        if destination == 'UNIQUAC' and (
            comp1 not in destination_thermo.r or comp2 not in destination_thermo.r
        ):
            continue
        low = float(options.get('Tmin_K', 0.0))
        high = float(options.get('Tmax_K', 0.0))
        if low <= 0.0 or high <= 0.0 or high <= low:
            failures.append(
                f"{comp1}/{comp2}: requires inherited or pair-specific "
                "0 < Tmin < Tmax"
            )
            continue
        source_name = options['source']
        source_class = _source_classes()[source_name]
        resolved_source_groups = {}
        unresolved_component = None
        unresolved_excluded = False
        for component in (comp1, comp2):
            group_key = (source_name, component)
            if group_key not in source_group_cache:
                source_group_cache[group_key] = _source_component_groups(
                    destination_thermo,
                    source_name,
                    source_class,
                    component,
                    unifac_groups,
                )
            groups, excluded = source_group_cache[group_key]
            if groups is None:
                unresolved_component = component
                unresolved_excluded = excluded
                break
            resolved_source_groups[component] = groups
        if unresolved_component is not None:
            exclusion_key = (source_name, unresolved_component)
            if unresolved_excluded:
                if exclusion_key not in reported_source_exclusions:
                    failures.append(
                        f"all pairs involving {unresolved_component}: {source_name} "
                        f"excludes {unresolved_component} from its liquid group model "
                        "(for example as a contextual Henry solute); no molecular "
                        "liquid interaction can be regressed"
                    )
                    reported_source_exclusions.add(exclusion_key)
            else:
                failures.append(
                    f"{comp1}/{comp2}: {source_name} groups not found for "
                    f"'{unresolved_component}'"
                )
            continue
        try:
            record, details = _fit_pair(
                destination_thermo,
                destination,
                comp1,
                comp2,
                options,
                unifac_groups,
                resolved_source_groups,
            )
        except Exception as error:
            failures.append(f"{comp1}/{comp2}: {error}")
            continue
        records.append(record)
        metadata[tuple(sorted((comp1, comp2)))] = details
        details['policy'] = policy

    if records:
        pairs = ', '.join(
            f"{record['component1']}/{record['component2']}" for record in records
        )
        worst_rmse = max(
            details['rmse_ln_gamma'] for details in metadata.values()
        )
        worst_maximum = max(
            details['max_abs_ln_gamma_error'] for details in metadata.values()
        )
        destination_thermo.add_warning(
            f"Estimated and froze {len(records)} {destination} binary "
            f"interaction(s) from UNIFAC-family activity coefficients: {pairs}. "
            f"Worst ln(gamma) RMSE={worst_rmse:.4g}, "
            f"max error={worst_maximum:.4g}."
        )
    for failure in failures:
        destination_thermo.add_warning(
            f"Could not estimate missing {destination} binary interaction for "
            f"{failure}; retaining the ideal residual-interaction fallback."
        )
    return records, metadata
