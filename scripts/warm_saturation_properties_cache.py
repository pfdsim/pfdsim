"""Warm resolved saturation-property rows from bundled local catalogs.

The candidate union includes the Perry 2-8/2-10 and CoolProp identities used
by the canonical-Psat warmer, every EOS-effective critical row, and every
compound in the ACS JCED 2015/IUPAC critical-property review extraction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import os
import sqlite3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_WORKER_DATABASE = None
_WORKER_RESOLVER = None


def source_candidates() -> list[dict]:
    from property_resolution.critical import (
        ACS_JCED_5B00571_TABLE1_PATH,
        EFFECTIVE_CRITICALS_DB_PATH,
    )
    from scripts.warm_canonical_psat_cache import source_candidates as psat_candidates
    from perry_properties import HEAT_OF_FUSION_DATA_PATH

    candidates: dict[str, dict] = {}

    def merge(
        cas,
        *,
        name='',
        formula='',
        MW=None,
        Tb=None,
        source,
    ) -> None:
        cas = str(cas or '').strip()
        if not cas:
            return
        item = candidates.setdefault(cas, {
            'cas': cas,
            'name': '',
            'formula': '',
            'MW': None,
            'Tb': None,
            'sources': set(),
        })
        if name and not item['name']:
            item['name'] = str(name)
        if formula and not item['formula']:
            item['formula'] = str(formula)
        if MW is not None and item['MW'] is None:
            item['MW'] = MW
        if Tb is not None and item['Tb'] is None:
            item['Tb'] = Tb
        item['sources'].add(str(source))

    for item in psat_candidates():
        for source in item['sources']:
            merge(
                item['cas'],
                name=item.get('name'),
                formula=item.get('formula'),
                Tb=item.get('Tb'),
                source=source,
            )

    if EFFECTIVE_CRITICALS_DB_PATH.exists():
        with sqlite3.connect(EFFECTIVE_CRITICALS_DB_PATH) as connection:
            for cas, name, MW in connection.execute(
                'SELECT CAS, name, MW FROM effective_criticals'
            ):
                merge(
                    cas,
                    name=name,
                    MW=MW,
                    source='effective_criticals.sqlite',
                )

    if ACS_JCED_5B00571_TABLE1_PATH.exists():
        payload = json.loads(ACS_JCED_5B00571_TABLE1_PATH.read_text())
        for cas, entry in (payload.get('chemicals') or {}).items():
            merge(
                cas,
                name=entry.get('name'),
                formula=entry.get('formula'),
                MW=entry.get('molar_mass_g_mol'),
                source='ACS JCED 2015/IUPAC Table 1',
            )

    if HEAT_OF_FUSION_DATA_PATH.exists():
        payload = json.loads(HEAT_OF_FUSION_DATA_PATH.read_text())
        for cas, entry in (payload.get('chemicals') or {}).items():
            fusion_rows = list(entry.get('heat_of_fusion') or [])
            first = fusion_rows[0] if fusion_rows else {}
            merge(
                cas,
                name=entry.get('name'),
                formula=entry.get('formula') or first.get('formula'),
                MW=first.get('molecular_weight_estimate'),
                source='Perry Table 2-68 fusion',
            )

    return sorted(
        (
            {
                **item,
                'name': item['name'] or item['cas'],
                'sources': tuple(sorted(item['sources'])),
            }
            for item in candidates.values()
        ),
        key=lambda item: item['cas'],
    )


def _initialize_worker() -> None:
    global _WORKER_DATABASE, _WORKER_RESOLVER
    from chemical_properties import ChemicalDatabase
    from property_resolver import get_property_resolver

    _WORKER_RESOLVER = get_property_resolver()
    _WORKER_DATABASE = ChemicalDatabase(enable_online=False)


def _minimal_props(candidate: dict) -> dict:
    props = {
        'symbol': candidate.get('formula') or candidate['name'],
        'name': candidate['name'],
        'formula': candidate.get('formula') or '',
        'CAS': candidate['cas'],
        'cas': candidate['cas'],
        'source': 'saturation_properties_cache_warm',
        'property_sources': {},
    }
    for field_name in ('MW', 'Tb'):
        value = candidate.get(field_name)
        if value is None:
            continue
        props[field_name] = value
        props['property_sources'][field_name] = {
            'source': 'local',
            'method': 'critical_cache_warm_catalog_identity',
            'quality': 0.95,
            'notes': 'Identity input assembled from bundled cache-warm catalogs',
        }
    return props


def _warm_one(candidate: dict) -> dict:
    started = time.perf_counter()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            component = _WORKER_DATABASE.get(
                candidate['cas'],
                fetch_online=False,
            )
            if component is None:
                props = _minimal_props(candidate)
                symbol = props['symbol']
            else:
                props = component.to_dict()
                symbol = component.symbol
            melting_result = _WORKER_RESOLVER.resolve_melting_point(
                symbol,
                props,
                allow_online=False,
            )
            if melting_result.value is not None:
                props['Tm'] = melting_result.value
                props.setdefault('property_sources', {})['Tm'] = {
                    'source': melting_result.source,
                    'method': melting_result.method,
                    'quality': melting_result.quality,
                    'notes': melting_result.notes,
                }
            fusion_records = _WORKER_RESOLVER.resolve_fusion_transitions(
                symbol,
                props,
                allow_online=False,
            )
            fusion_result = _WORKER_RESOLVER.resolve_hfus(
                symbol,
                props,
                allow_online=False,
            )
            triple_results = _WORKER_RESOLVER.resolve_triple_point(
                symbol,
                props,
                allow_online=False,
            )
            for field_name, result in triple_results.items():
                if result.value is None:
                    continue
                props[field_name] = result.value
                props.setdefault('property_sources', {})[field_name] = {
                    'source': result.source,
                    'method': result.method,
                    'quality': result.quality,
                    'notes': result.notes,
                }
            boiling_result = _WORKER_RESOLVER.resolve_boiling_point(
                symbol,
                props,
                allow_online=False,
                allow_estimation=True,
            )
            if boiling_result.value is not None:
                props['Tb'] = boiling_result.value
                props.setdefault('property_sources', {})['Tb'] = {
                    'source': boiling_result.source,
                    'method': boiling_result.method,
                    'quality': boiling_result.quality,
                    'notes': boiling_result.notes,
                }
            elif boiling_result.method in {
                'no_normal_boiling_point_at_1atm',
                'invalid_normal_boiling_point_below_triple_point',
            }:
                props.pop('Tb', None)
                props.setdefault('property_sources', {}).pop('Tb', None)
            critical_results = _WORKER_RESOLVER.resolve_critical_properties(
                symbol,
                props,
                allow_online=False,
                allow_estimation=True,
            )
        critical_resolved = {
            name: result
            for name, result in critical_results.items()
            if result.value is not None
        }
        triple_resolved = {
            name: result
            for name, result in triple_results.items()
            if result.value is not None
        }
        return {
            'ok': True,
            'critical_complete': len(critical_resolved) == 5,
            'triple_complete': len(triple_resolved) == 2,
            'boiling_complete': boiling_result.value is not None,
            'melting_complete': melting_result.value is not None,
            'fusion_complete': fusion_result.value is not None,
            'fusion_record_count': len(fusion_records),
            'cas': candidate['cas'],
            'name': candidate['name'],
            'sources': candidate['sources'],
            'elapsed': time.perf_counter() - started,
            'critical_resolved_count': len(critical_resolved),
            'triple_resolved_count': len(triple_resolved),
            'methods': {
                'critical': {
                    name: result.method
                    for name, result in critical_results.items()
                },
                'triple': {
                    name: result.method
                    for name, result in triple_results.items()
                },
                'boiling': boiling_result.method,
                'melting': melting_result.method,
                'fusion': fusion_result.method,
            },
        }
    except Exception as error:
        return {
            'ok': False,
            'critical_complete': False,
            'triple_complete': False,
            'boiling_complete': False,
            'melting_complete': False,
            'fusion_complete': False,
            'fusion_record_count': 0,
            'cas': candidate['cas'],
            'name': candidate['name'],
            'sources': candidate['sources'],
            'elapsed': time.perf_counter() - started,
            'error': f'{type(error).__name__}: {error}',
            'captured_output': output.getvalue()[-2000:],
        }


def _warm_fusion_one(candidate: dict) -> dict:
    """Warm only Tm-dependent fusion records and their selected scalar."""
    started = time.perf_counter()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            component = _WORKER_DATABASE.get(
                candidate['cas'],
                fetch_online=False,
            )
            if component is None:
                props = _minimal_props(candidate)
                symbol = props['symbol']
            else:
                props = component.to_dict()
                symbol = component.symbol
            melting_result = _WORKER_RESOLVER.resolve_melting_point(
                symbol,
                props,
                allow_online=False,
            )
            if melting_result.value is not None:
                props['Tm'] = melting_result.value
                props.setdefault('property_sources', {})['Tm'] = {
                    'source': melting_result.source,
                    'method': melting_result.method,
                    'quality': melting_result.quality,
                    'notes': melting_result.notes,
                }
            fusion_records = _WORKER_RESOLVER.resolve_fusion_transitions(
                symbol,
                props,
                allow_online=False,
            )
            fusion_result = _WORKER_RESOLVER.resolve_hfus(
                symbol,
                props,
                allow_online=False,
            )
        return {
            'ok': True,
            'critical_complete': False,
            'triple_complete': False,
            'boiling_complete': False,
            'melting_complete': melting_result.value is not None,
            'fusion_complete': fusion_result.value is not None,
            'fusion_record_count': len(fusion_records),
            'cas': candidate['cas'],
            'name': candidate['name'],
            'sources': candidate['sources'],
            'elapsed': time.perf_counter() - started,
            'methods': {
                'melting': melting_result.method,
                'fusion': fusion_result.method,
            },
        }
    except Exception as error:
        return {
            'ok': False,
            'critical_complete': False,
            'triple_complete': False,
            'boiling_complete': False,
            'melting_complete': False,
            'fusion_complete': False,
            'fusion_record_count': 0,
            'cas': candidate['cas'],
            'name': candidate['name'],
            'sources': candidate['sources'],
            'elapsed': time.perf_counter() - started,
            'error': f'{type(error).__name__}: {error}',
            'captured_output': output.getvalue()[-2000:],
        }


def _cache_row_counts(
    path: Path,
    critical_version: int,
    phase_version: int,
    fusion_version: int,
    online_phase_version: int,
) -> dict:
    with sqlite3.connect(path) as connection:
        return {
            'critical': int(connection.execute(
                """
                SELECT count(*) FROM resolved_critical_properties_cache
                WHERE cache_version = ?
                """,
                (critical_version,),
            ).fetchone()[0]),
            'triple_point': int(connection.execute(
                """
                SELECT count(*) FROM resolved_phase_point_cache
                WHERE cache_version = ? AND resolution_kind = 'triple_point'
                """,
                (phase_version,),
            ).fetchone()[0]),
            'boiling_point': int(connection.execute(
                """
                SELECT count(*) FROM resolved_phase_point_cache
                WHERE cache_version = ? AND resolution_kind = 'boiling_point'
                """,
                (phase_version,),
            ).fetchone()[0]),
            'melting_point': int(connection.execute(
                """
                SELECT count(*) FROM resolved_phase_point_cache
                WHERE cache_version = ? AND resolution_kind = 'melting_point'
                """,
                (phase_version,),
            ).fetchone()[0]),
            'fusion_transition_rows': int(connection.execute(
                """
                SELECT count(*) FROM resolved_fusion_transition_cache
                WHERE cache_version = ?
                  AND online_phase_contract_version = ?
                  AND allow_online = 0
                """,
                (fusion_version, online_phase_version),
            ).fetchone()[0]),
            'fusion_transition_components': int(connection.execute(
                """
                SELECT count(DISTINCT component_key)
                FROM resolved_fusion_transition_cache
                WHERE cache_version = ?
                  AND online_phase_contract_version = ?
                  AND allow_online = 0
                """,
                (fusion_version, online_phase_version),
            ).fetchone()[0]),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workers',
        type=int,
        default=min(6, os.cpu_count() or 1),
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=Path('/tmp/saturation_properties_cache_warm_report.json'),
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit nonzero when a candidate raises during cache warming.',
    )
    parser.add_argument(
        '--only',
        choices=('all', 'fusion'),
        default='all',
        help=(
            'Warm the full saturation-property bundle or only the '
            'Tm-dependent fusion-transition cache.'
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')

    from property_resolution.resolver import PropertyResolver

    resolver = PropertyResolver()
    cache_path = resolver.initialize_critical_properties_disk_cache()
    resolver.initialize_phase_point_disk_cache()
    candidates = source_candidates()
    started = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context('fork'),
        initializer=_initialize_worker,
    ) as pool:
        worker = _warm_fusion_one if args.only == 'fusion' else _warm_one
        futures = [pool.submit(worker, item) for item in candidates]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                cached = sum(item['ok'] for item in results)
                critical_complete = sum(
                    item['critical_complete'] for item in results
                )
                triple_complete = sum(
                    item['triple_complete'] for item in results
                )
                boiling_complete = sum(
                    item['boiling_complete'] for item in results
                )
                melting_complete = sum(
                    item['melting_complete'] for item in results
                )
                fusion_complete = sum(
                    item['fusion_complete'] for item in results
                )
                if args.only == 'fusion':
                    print(
                        f'{index}/{len(futures)} processed; '
                        f'{cached} cached, '
                        f'{fusion_complete} fusion complete, '
                        f'{index - cached} errors',
                        flush=True,
                    )
                else:
                    print(
                        f'{index}/{len(futures)} processed; '
                        f'{cached} cached, '
                        f'{critical_complete} critical complete, '
                        f'{triple_complete} triple complete, '
                        f'{boiling_complete} boiling complete, '
                        f'{melting_complete} melting complete, '
                        f'{fusion_complete} fusion complete, '
                        f'{index - cached} errors',
                        flush=True,
                    )

    results.sort(key=lambda item: item['cas'])
    source_counts = {}
    for candidate in candidates:
        for source in candidate['sources']:
            source_counts[source] = source_counts.get(source, 0) + 1
    payload = {
        'cache_path': str(cache_path),
        'critical_cache_version': resolver.CRITICAL_PROPERTIES_CACHE_VERSION,
        'phase_point_cache_version': resolver.PHASE_POINT_CACHE_VERSION,
        'fusion_transition_cache_version': (
            resolver.FUSION_TRANSITION_CACHE_VERSION
        ),
        'online_phase_contract_version': (
            resolver.ONLINE_PHASE_CHANGE_CACHE_VERSION
        ),
        'workers': args.workers,
        'only': args.only,
        'elapsed_seconds': time.perf_counter() - started,
        'candidate_count': len(candidates),
        'source_candidate_counts': source_counts,
        'success_count': sum(item['ok'] for item in results),
        'critical_complete_count': sum(
            item['critical_complete'] for item in results
        ),
        'triple_complete_count': sum(
            item['triple_complete'] for item in results
        ),
        'boiling_complete_count': sum(
            item['boiling_complete'] for item in results
        ),
        'melting_complete_count': sum(
            item['melting_complete'] for item in results
        ),
        'fusion_complete_count': sum(
            item['fusion_complete'] for item in results
        ),
        'fusion_record_count': sum(
            item['fusion_record_count'] for item in results
        ),
        'failure_count': sum(not item['ok'] for item in results),
        'database_row_counts': _cache_row_counts(
            cache_path,
            resolver.CRITICAL_PROPERTIES_CACHE_VERSION,
            resolver.PHASE_POINT_CACHE_VERSION,
            resolver.FUSION_TRANSITION_CACHE_VERSION,
            resolver.ONLINE_PHASE_CHANGE_CACHE_VERSION,
        ),
        'results': results,
    }
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"Cache rows: {payload['database_row_counts']}; "
        f'report: {args.report}'
    )
    return 1 if args.strict and payload['failure_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
