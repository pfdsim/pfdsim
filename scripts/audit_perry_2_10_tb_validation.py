#!/usr/bin/env python3
"""Audit every Perry Table 2-10 row against the shared hard-Tb gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.common import validate_psat_boiling_point
from property_resolution.resolver import PropertyResolver
from property_resolution.vapor_pressure_adapter import (
    PsatCanonicalizationAdapter,
    _qualified_boiling_point_candidate,
)
from perry_properties import get_perry_property_library


PERRY_PATH = ROOT / 'data' / 'perry_table_2_10_vapor_pressure.json'
SATURATION_CACHE_PATH = ROOT / 'data' / 'runtime' / 'saturation_properties_cache.sqlite'
CANONICAL_CACHE_PATH = ROOT / 'data' / 'runtime' / 'canonical_psat_cache.sqlite'
DEFAULT_REPORT_PATH = Path('/tmp/perry_2_10_tb_validation_report.json')


def _cached_phase_payload(resolver, provider: str, identifier: str):
    version = resolver.ONLINE_PHASE_CHANGE_CACHE_VERSION
    payload = resolver._get_cache(f'phase_{provider}_v{version}_{identifier}')
    if resolver._is_missing_cache(payload):
        return None
    return payload


def _qualified_tb(resolver, entry: dict) -> tuple[object, bool]:
    props = {
        'symbol': entry.get('formula') or entry.get('name'),
        'name': entry.get('name'),
        'CAS': entry.get('cas'),
        'formula': entry.get('formula'),
    }
    triple = resolver.resolve_triple_point(
        str(props['symbol']),
        props,
        allow_online=True,
    )
    property_sources = {}
    for name in ('Tt', 'Pt'):
        item = triple.get(name)
        if item is None or item.value is None:
            continue
        props[name] = item.value
        property_sources[name] = {
            'source': item.source,
            'method': item.method,
            'quality': item.quality,
            'notes': item.notes,
        }
    if property_sources:
        props['property_sources'] = property_sources
    result = resolver.resolve_boiling_point(
        str(props['symbol']),
        props,
        allow_online=True,
        allow_estimation=False,
    )
    if result.value is None:
        return None, False
    component = dict(props)
    component['Tb'] = result.value
    component['property_sources'] = {
        **property_sources,
        'Tb': {
            'source': result.source,
            'method': result.method,
            'quality': result.quality,
            'notes': result.notes,
        },
    }
    candidate = _qualified_boiling_point_candidate(
        PsatCanonicalizationAdapter(component),
        explicit_value=float(result.value),
    )
    online_used = (
        str(result.source).lower() == 'online'
        or str(result.method).lower() in {'pubchem', 'nist_phase_change'}
    )
    return candidate, online_used


def _population_snapshot(
    *,
    saturation_cache_path: Path = SATURATION_CACHE_PATH,
    canonical_cache_path: Path = CANONICAL_CACHE_PATH,
) -> dict:
    snapshot = {
        'canonical_coverage': 0,
        'canonical_version': None,
        'canonical_source_method_counts': {},
        'perry_2_10_selected_intervals': 0,
        'perry_2_10_quality_basis_counts': {},
        'psat_derived_omega_count': 0,
        'critical_version': None,
        'psat_derived_omega_by_cas': {},
        'canonical_component_keys': [],
    }
    if canonical_cache_path.exists():
        with sqlite3.connect(canonical_cache_path) as connection:
            connection.row_factory = sqlite3.Row
            version_row = connection.execute(
                'SELECT cache_version FROM canonical_psat_cache '
                'GROUP BY cache_version ORDER BY count(*) DESC, '
                'cache_version DESC LIMIT 1'
            ).fetchone()
            version = None if version_row is None else version_row[0]
            snapshot['canonical_version'] = version
            if version is not None:
                rows = connection.execute(
                    'SELECT component_key, provenance_json '
                    'FROM canonical_psat_cache WHERE cache_version = ?',
                    (version,),
                ).fetchall()
                methods = Counter()
                bases = Counter()
                perry_count = 0
                for row in rows:
                    for item in json.loads(row['provenance_json']):
                        method = str(item.get('method') or '')
                        methods[method] += 1
                        if method == 'perry_2_10_vapor_pressure':
                            perry_count += 1
                            bases[str(
                                (item.get('metadata') or {}).get('quality_basis')
                                or 'unspecified'
                            )] += 1
                snapshot.update({
                    'canonical_coverage': len(rows),
                    'canonical_source_method_counts': dict(sorted(methods.items())),
                    'perry_2_10_selected_intervals': perry_count,
                    'perry_2_10_quality_basis_counts': dict(sorted(bases.items())),
                    'canonical_component_keys': sorted({
                        str(row['component_key']) for row in rows
                    }),
                })
    if saturation_cache_path.exists():
        with sqlite3.connect(saturation_cache_path) as connection:
            connection.row_factory = sqlite3.Row
            version_row = connection.execute(
                'SELECT cache_version FROM resolved_critical_properties_cache '
                'GROUP BY cache_version ORDER BY count(*) DESC, '
                'cache_version DESC LIMIT 1'
            ).fetchone()
            version = None if version_row is None else version_row[0]
            snapshot['critical_version'] = version
            if version is not None:
                rows = connection.execute(
                    'SELECT cas, omega_value, omega_quality '
                    'FROM resolved_critical_properties_cache '
                    'WHERE cache_version = ? '
                    "AND omega_method = 'psat_definition_at_Tr_0_7'",
                    (version,),
                ).fetchall()
                by_cas = {}
                for row in rows:
                    cas = str(row['cas'] or '')
                    if not cas:
                        continue
                    current = by_cas.get(cas)
                    candidate = {
                        'value': row['omega_value'],
                        'quality': row['omega_quality'],
                    }
                    if current is None or candidate['quality'] > current['quality']:
                        by_cas[cas] = candidate
                snapshot['psat_derived_omega_count'] = len(by_cas)
                snapshot['psat_derived_omega_by_cas'] = dict(sorted(by_cas.items()))
    return snapshot


def _population_comparison(before: dict, after: dict) -> dict:
    before_keys = set(before.get('canonical_component_keys') or ())
    after_keys = set(after.get('canonical_component_keys') or ())
    before_omega = before.get('psat_derived_omega_by_cas') or {}
    after_omega = after.get('psat_derived_omega_by_cas') or {}
    changes = []
    for cas in sorted(set(before_omega) | set(after_omega)):
        old = before_omega.get(cas)
        new = after_omega.get(cas)
        old_value = None if old is None else old.get('value')
        new_value = None if new is None else new.get('value')
        if old_value == new_value:
            continue
        changes.append({
            'cas': cas,
            'before': old,
            'after': new,
            'delta': (
                None
                if old_value is None or new_value is None
                else new_value - old_value
            ),
        })
    changes.sort(
        key=lambda item: abs(item['delta']) if item['delta'] is not None else float('inf'),
        reverse=True,
    )
    return {
        'canonical_coverage_delta': (
            after.get('canonical_coverage', 0)
            - before.get('canonical_coverage', 0)
        ),
        'newly_unavailable_canonical_components': sorted(before_keys - after_keys),
        'newly_available_canonical_components': sorted(after_keys - before_keys),
        'psat_derived_omega_count_delta': (
            after.get('psat_derived_omega_count', 0)
            - before.get('psat_derived_omega_count', 0)
        ),
        'largest_omega_changes': changes[:50],
    }


def build_report(
    existing: dict | None = None,
    before_population: dict | None = None,
) -> dict:
    payload = json.loads(PERRY_PATH.read_text())
    library = get_perry_property_library()
    records = []
    with tempfile.TemporaryDirectory(prefix='pfdsim-perry-tb-audit-') as directory:
        resolver = PropertyResolver()
        resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(directory) / 'selected.sqlite'
        resolver._fetch_phase_change_pubchem = lambda identifier: (
            _cached_phase_payload(resolver, 'pubchem', str(identifier))
        )
        resolver._fetch_phase_change_nist = lambda identifier: (
            _cached_phase_payload(resolver, 'nist', str(identifier))
        )
        for cas, entry in sorted(payload['chemicals'].items()):
            curve = library.table_2_10_vapor_pressure_curve(cas)
            if curve is None:
                raise RuntimeError(f'Missing Perry 2-10 curve for {cas}')
            interpolator, row = curve
            T_min = float(row['T_min_K'])
            T_max = float(row['T_max_K'])
            candidate, cached_online = _qualified_tb(resolver, entry)
            record = {
                'cas': cas,
                'name': entry.get('name'),
                'qualified_Tb_K': None,
                'Tb_source': None,
                'Tb_method': None,
                'Tb_quality': None,
                'cached_online_Tb_used': bool(cached_online),
                'source_T_min_K': T_min,
                'source_T_max_K': T_max,
                'Psat_at_Tb_bar': None,
                'relative_error': None,
                'status': 'unavailable',
            }
            if candidate is not None:
                Tb = float(candidate.value)
                record.update({
                    'qualified_Tb_K': Tb,
                    'Tb_source': candidate.source,
                    'Tb_method': candidate.method,
                    'Tb_quality': candidate.quality,
                })

                def pressure_at_temperature(temperature):
                    return float(math.exp(float(interpolator(temperature))))

                validation = validate_psat_boiling_point(
                    pressure_at_temperature,
                    T_min,
                    T_max,
                    Tb,
                )
                record['Psat_at_Tb_bar'] = validation.pressure_bar
                record['relative_error'] = validation.relative_error
                if not validation.covers_boiling_point:
                    record['status'] = 'outside_range'
                elif validation.accepted:
                    record['status'] = 'accepted'
                else:
                    record['status'] = 'rejected'
            records.append(record)

    statuses = Counter(record['status'] for record in records)
    in_range = [
        record for record in records
        if record['status'] in {'accepted', 'rejected'}
    ]
    thresholds = (0.02, 0.03, 0.05, 0.10, 0.20, 0.50)
    aggregates = {
        'total_rows': len(records),
        'rows_with_qualified_Tb': sum(
            record['qualified_Tb_K'] is not None for record in records
        ),
        'qualified_Tb_inside_source_range': len(in_range),
        'accepted_count': statuses['accepted'],
        'rejected_count': statuses['rejected'],
        'outside_range_count': statuses['outside_range'],
        'unavailable_count': statuses['unavailable'],
        'rejection_rate_among_in_range_qualified': (
            statuses['rejected'] / len(in_range) if in_range else 0.0
        ),
        'rejection_rate_across_all_rows': statuses['rejected'] / len(records),
        'relative_error_threshold_counts': {
            f'above_{int(threshold * 100)}_percent': sum(
                record['relative_error'] is not None
                and record['relative_error'] > threshold
                for record in records
            )
            for threshold in thresholds
        },
    }
    current_population = _population_snapshot()
    report = {
        'contract': 'perry_2_10_shared_tb_validation_v1',
        'aggregates': aggregates,
        'records': records,
    }
    if before_population is not None:
        report['population_before'] = before_population
        report['population_after'] = current_population
        report['population_comparison'] = _population_comparison(
            before_population,
            current_population,
        )
    elif existing and existing.get('population_before'):
        report['population_before'] = existing['population_before']
        report['population_after'] = current_population
        report['population_comparison'] = _population_comparison(
            existing['population_before'],
            current_population,
        )
    else:
        report['population_before'] = current_population
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument('--before-saturation-cache', type=Path)
    parser.add_argument('--before-canonical-cache', type=Path)
    args = parser.parse_args()
    existing = None
    if args.output.exists():
        existing = json.loads(args.output.read_text())
    before_population = None
    if args.before_saturation_cache or args.before_canonical_cache:
        if not args.before_saturation_cache or not args.before_canonical_cache:
            parser.error('both before-cache paths are required together')
        before_population = _population_snapshot(
            saturation_cache_path=args.before_saturation_cache,
            canonical_cache_path=args.before_canonical_cache,
        )
    report = build_report(existing, before_population)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['aggregates'], indent=2, sort_keys=True))
    if 'population_comparison' in report:
        print(json.dumps(report['population_comparison'], indent=2, sort_keys=True))
    print(f'report: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
