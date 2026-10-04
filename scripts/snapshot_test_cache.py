"""Export current, positively identified provider records for offline tests.

This is an explicit fixture maintenance command, never part of test startup.
It reads a supplied SQLite cache in read-only mode, excludes selected/derived
results and unknown identifiers, and writes a reviewable source-data snapshot.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compound_identity import CompoundIdentityResolver


def snapshot(source):
    chemicals = json.loads((ROOT/'data'/'chemicals.json').read_text())['chemicals']
    identifiers = set()
    for symbol, props in chemicals.items():
        for value in (symbol, props.get('name'), props.get('CAS'), *props.get('aliases', [])):
            if isinstance(value, str) and value:
                identifiers.add(value.casefold())
    identifiers.update(alias.casefold() for alias, symbol in
                       CompoundIdentityResolver.MANUAL_ALIASES.items() if symbol in chemicals)
    families = {
        'property_resolver': (
            'phase_pubchem_v9_', 'phase_nist_v9_', 'cp_nist_v3_',
            'nist_antoine_rows_', 'formation_nist_', 'pubchem_cid_',
            'density_pubchem_v2_', 'viscosity_pubchem_v2_',
        ),
        'online_property_fetcher': (
            'pubchem_component_v6_', 'pubchem_component_cas_v6_',
            'pubchem_structure_',
        ),
    }
    grouped = {}
    with closing(sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro', uri=True)) as connection:
        rows = connection.execute(
            'SELECT namespace, cache_key, payload_json, updated_at_utc '
            'FROM runtime_json_cache WHERE is_missing=0 ORDER BY namespace, cache_key'
        )
        for namespace, key, payload_json, captured_at in rows:
            prefix = next((value for value in families.get(namespace, ())
                           if key.startswith(value)), None)
            if prefix is None or key[len(prefix):].casefold() not in identifiers:
                continue
            payload = json.loads(payload_json)
            if payload.get('_missing'):
                continue
            identity = namespace, json.dumps(payload, sort_keys=True)
            record = grouped.setdefault(identity, {
                'namespace': namespace, 'keys': [], 'payload': payload,
                'captured_at_utc': captured_at,
            })
            record['keys'].append(key)
    return {
        'version': 1,
        'description': 'Frozen positive provider source records for known local compounds; '
                       'no derived answers, negative lookups, QM artifacts, or personal aliases.',
        'records': list(grouped.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = snapshot(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    print(f"Wrote {len(data['records'])} source records with "
          f"{sum(len(record['keys']) for record in data['records'])} keys")


if __name__ == '__main__':
    main()
