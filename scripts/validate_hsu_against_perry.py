#!/usr/bin/env python3
"""Validate Hsu liquid viscosity against Perry correlations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import get_perry_property_library
from property_resolver import PropertyResolver

# The resolver's Hsu rung now lives in hsu_method.py, which carries
# per-molecule validity windows (fragmentation.tr_min/tr_max); this global
# cap matches its default upper limit.  See also scripts/hsu/perry_benchmark.py
# for the native-engine acceptance benchmark.
HSU_MAX_REDUCED_TEMPERATURE = 0.75


SAMPLE_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--chemicals-path',
        type=Path,
        help='Directory containing a temporary installation of the chemicals package.',
    )
    parser.add_argument('--json', type=Path, help='Optional path for complete JSON results.')
    parser.add_argument('--worst', type=int, default=20, help='Worst compounds to print.')
    return parser.parse_args()


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for low, high in sorted(intervals):
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            continue
        if not merged or low > merged[-1][1]:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return [(low, high) for low, high in merged]


def sample_intervals(intervals: list[tuple[float, float]]) -> list[float]:
    intervals = merge_intervals(intervals)
    if not intervals:
        return []
    lengths = [max(high - low, 0.0) for low, high in intervals]
    total = sum(lengths)
    if total <= 0.0:
        return [intervals[0][0]] * len(SAMPLE_QUANTILES)

    samples = []
    for quantile in SAMPLE_QUANTILES:
        target = quantile * total
        traversed = 0.0
        for (low, high), length in zip(intervals, lengths):
            if target <= traversed + length or (low, high) == intervals[-1]:
                samples.append(low + max(0.0, target - traversed))
                break
            traversed += length
    return samples


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def critical_props(entry: dict, smiles: str) -> dict | None:
    critical = entry.get('critical_constants') or {}
    required = ('Tc_K', 'Pc_MPa', 'Vc_m3_per_kmol', 'Zc', 'omega')
    if any(critical.get(key) is None for key in required):
        return None
    viscosity_rows = entry.get('liquid_viscosity') or []
    molecular_weight = next(
        (row.get('molecular_weight') for row in viscosity_rows if row.get('molecular_weight')),
        None,
    )
    if molecular_weight is None:
        return None
    return {
        'CAS': entry.get('cas'),
        'name': entry.get('name'),
        'formula': entry.get('formula'),
        'smiles': smiles,
        'MW': molecular_weight,
        'Tc': critical['Tc_K'],
        'Pc': critical['Pc_MPa'] * 10.0,
        'Vc': critical['Vc_m3_per_kmol'] * 1000.0,
        'Zc': critical['Zc'],
        'omega': critical['omega'],
        '_allow_online_lookup': False,
    }


def resolve_smiles_text(identifier: str) -> str | None:
    try:
        from chemical_properties import ChemicalDatabase
        result = ChemicalDatabase(enable_online=False).resolve_smiles_info(
            identifier,
            fetch_online=False,
        )
        return result.smiles if result else None
    except Exception:
        return None


def supported_intervals(entry: dict, upper: float | None = None) -> list[tuple[float, float]]:
    intervals = []
    for row in entry.get('liquid_viscosity') or []:
        if row.get('equation_id') not in {100, 101}:
            continue
        low = float(row.get('T_min_K', -math.inf))
        high = float(row.get('T_max_K', math.inf))
        if upper is not None:
            high = min(high, upper)
        if high >= low:
            intervals.append((low, high))
    return merge_intervals(intervals)


def summarize(records: list[dict]) -> dict:
    if not records:
        return {
            'compounds': 0,
            'points': 0,
        }
    by_cas: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_cas[record['cas']].append(record)
    absolute_errors = [record['absolute_percent_error'] for record in records]
    signed_errors = [record['signed_percent_error'] for record in records]
    compound_mapes = [
        statistics.fmean(item['absolute_percent_error'] for item in items)
        for items in by_cas.values()
    ]
    compound_biases = [
        statistics.fmean(item['signed_percent_error'] for item in items)
        for items in by_cas.values()
    ]
    return {
        'compounds': len(by_cas),
        'points': len(records),
        'compound_balanced_mape_percent': statistics.fmean(compound_mapes),
        'compound_balanced_bias_percent': statistics.fmean(compound_biases),
        'point_mape_percent': statistics.fmean(absolute_errors),
        'point_bias_percent': statistics.fmean(signed_errors),
        'median_absolute_percent_error': statistics.median(absolute_errors),
        'p90_absolute_percent_error': percentile(absolute_errors, 0.90),
        'p95_absolute_percent_error': percentile(absolute_errors, 0.95),
        'within_10_percent_fraction': sum(value <= 10.0 for value in absolute_errors) / len(absolute_errors),
        'within_20_percent_fraction': sum(value <= 20.0 for value in absolute_errors) / len(absolute_errors),
        'within_30_percent_fraction': sum(value <= 30.0 for value in absolute_errors) / len(absolute_errors),
        'within_50_percent_fraction': sum(value <= 50.0 for value in absolute_errors) / len(absolute_errors),
    }


def worst_compounds(records: list[dict]) -> list[dict]:
    by_cas: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_cas[record['cas']].append(record)
    worst = []
    for cas, items in by_cas.items():
        worst.append({
            'cas': cas,
            'name': items[0]['name'],
            'formula': items[0]['formula'],
            'points': len(items),
            'mape_percent': statistics.fmean(item['absolute_percent_error'] for item in items),
            'bias_percent': statistics.fmean(item['signed_percent_error'] for item in items),
            'max_absolute_percent_error': max(item['absolute_percent_error'] for item in items),
        })
    return sorted(worst, key=lambda item: item['mape_percent'], reverse=True)


def evaluate_grid(
    resolver: PropertyResolver,
    library,
    cas: str,
    entry: dict,
    props: dict,
    temperatures: list[float],
) -> tuple[list[dict], Counter]:
    records = []
    failures = Counter()
    for temperature in temperatures:
        perry = library.viscosity_Pa_s(cas, temperature, 'liquid')
        if perry is None or perry.value <= 0.0:
            failures['perry_evaluation_failed'] += 1
            continue
        hsu = resolver._hsu_liquid_viscosity(cas, props, temperature, 'liquid')
        if hsu is None or hsu.value <= 0.0:
            if temperature / float(props['Tc']) >= HSU_MAX_REDUCED_TEMPERATURE:
                failures['outside_hsu_temperature_domain'] += 1
            else:
                failures['hsu_evaluation_failed'] += 1
            continue
        signed = 100.0 * (hsu.value / perry.value - 1.0)
        records.append({
            'cas': cas,
            'name': entry.get('name'),
            'formula': entry.get('formula'),
            'smiles': props.get('smiles'),
            'temperature_K': temperature,
            'reduced_temperature': temperature / float(props['Tc']),
            'perry_viscosity_Pa_s': perry.value,
            'hsu_viscosity_Pa_s': hsu.value,
            'signed_percent_error': signed,
            'absolute_percent_error': abs(signed),
            'hsu_quality': hsu.quality,
            'hsu_notes': hsu.notes,
        })
    return records, failures


def main() -> int:
    args = parse_args()
    if args.chemicals_path:
        sys.path.insert(0, str(args.chemicals_path.resolve()))
    try:
        import chemicals
        from chemicals.identifiers import search_chemical
    except ImportError as exc:
        raise SystemExit(
            'Install chemicals into a temporary directory and pass --chemicals-path.'
        ) from exc

    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass

    library = get_perry_property_library()
    library._load()
    resolver = PropertyResolver()

    result = {
        'chemicals_version': chemicals.__version__,
        'sample_quantiles': SAMPLE_QUANTILES,
        'hsu_max_reduced_temperature': HSU_MAX_REDUCED_TEMPERATURE,
        'counts': Counter(),
        'failures': Counter(),
        'failure_compounds': [],
        'perry_grid_records': [],
        'hsu_overlap_records': [],
    }

    for cas, entry in sorted(library.chemicals.items()):
        intervals = supported_intervals(entry)
        if not intervals:
            continue
        result['counts']['perry_liquid_viscosity_compounds'] += 1

        smiles = resolve_smiles_text(cas)
        if not smiles:
            result['failures']['cas_to_smiles_failed'] += 1
            result['failure_compounds'].append({
                'cas': cas, 'name': entry.get('name'), 'reason': 'cas_to_smiles_failed',
            })
            continue
        result['counts']['smiles_resolved_compounds'] += 1

        props = critical_props(entry, smiles)
        if props is None:
            result['failures']['critical_or_mw_missing'] += 1
            result['failure_compounds'].append({
                'cas': cas, 'name': entry.get('name'), 'reason': 'critical_or_mw_missing',
            })
            continue
        result['counts']['critical_ready_compounds'] += 1

        if resolver._hsu_fragmentation(cas, props) is None:
            result['failures']['hsu_fragmentation_failed'] += 1
            result['failure_compounds'].append({
                'cas': cas, 'name': entry.get('name'), 'reason': 'hsu_fragmentation_failed',
            })
            continue
        result['counts']['hsu_fragmentable_compounds'] += 1

        perry_temperatures = sample_intervals(intervals)
        records, failures = evaluate_grid(
            resolver, library, cas, entry, props, perry_temperatures,
        )
        result['perry_grid_records'].extend(records)
        result['failures'].update({f'perry_grid_{key}': value for key, value in failures.items()})

        hsu_upper = HSU_MAX_REDUCED_TEMPERATURE * float(props['Tc']) * (1.0 - 1.0e-12)
        overlap_intervals = supported_intervals(entry, upper=hsu_upper)
        if not overlap_intervals:
            result['failures']['no_perry_hsu_temperature_overlap'] += 1
            result['failure_compounds'].append({
                'cas': cas,
                'name': entry.get('name'),
                'reason': 'no_perry_hsu_temperature_overlap',
            })
            continue
        overlap_temperatures = sample_intervals(overlap_intervals)
        records, failures = evaluate_grid(
            resolver, library, cas, entry, props, overlap_temperatures,
        )
        result['hsu_overlap_records'].extend(records)
        result['failures'].update({f'overlap_grid_{key}': value for key, value in failures.items()})

    result['counts'] = dict(result['counts'])
    result['failures'] = dict(result['failures'])
    result['perry_grid_summary'] = summarize(result['perry_grid_records'])
    result['hsu_overlap_summary'] = summarize(result['hsu_overlap_records'])
    result['perry_grid_worst_compounds'] = worst_compounds(result['perry_grid_records'])
    result['hsu_overlap_worst_compounds'] = worst_compounds(result['hsu_overlap_records'])

    print(f"chemicals version: {result['chemicals_version']}")
    print('counts:', json.dumps(result['counts'], sort_keys=True))
    print('failures:', json.dumps(result['failures'], sort_keys=True))
    for label, key in (
        ('Literal Perry-range grid', 'perry_grid_summary'),
        ('Perry/Hsu overlap grid', 'hsu_overlap_summary'),
    ):
        print(f'\n{label}:')
        print(json.dumps(result[key], indent=2, sort_keys=True))

    print(f'\nWorst {args.worst} compounds on Perry/Hsu overlap grid:')
    for item in result['hsu_overlap_worst_compounds'][:args.worst]:
        print(
            f"{item['cas']:>12}  {item['mape_percent']:9.2f}% MAPE  "
            f"{item['bias_percent']:+9.2f}% bias  {item['name']} ({item['formula']})"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(f'\nWrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
