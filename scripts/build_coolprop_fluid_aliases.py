"""Build the bundled CoolProp fluid alias map used for cheap preflight lookup."""

from __future__ import annotations

import json
from pathlib import Path

import CoolProp.CoolProp as CP


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "coolprop_fluid_aliases.json"


def coolprop_key(identifier: str) -> str:
    return ''.join(ch.lower() for ch in str(identifier) if ch.isalnum())


def main() -> None:
    alias_candidates: dict[str, set[str]] = {}
    fluids = sorted(CP.get_global_param_string('FluidsList').split(','))
    for fluid in fluids:
        candidates = [fluid]
        for key in ('CAS', 'REFPROP_name', 'aliases'):
            try:
                value = CP.get_fluid_param_string(fluid, key)
            except Exception:
                value = ''
            if not value:
                continue
            if key == 'aliases':
                candidates.extend(part.strip() for part in value.split(',') if part.strip())
            else:
                candidates.append(value)

        for candidate in candidates:
            normalized = coolprop_key(candidate)
            if normalized:
                alias_candidates.setdefault(normalized, set()).add(fluid)

    ambiguous_aliases = {
        alias: sorted(matches)
        for alias, matches in alias_candidates.items()
        if len(matches) > 1
    }
    aliases = {
        alias: next(iter(matches))
        for alias, matches in alias_candidates.items()
        if len(matches) == 1
    }

    payload = {
        'source': 'CoolProp',
        'coolprop_version': CP.get_global_param_string('version'),
        'fluid_count': len(fluids),
        'aliases': dict(sorted(aliases.items())),
        'ambiguous_aliases': dict(sorted(ambiguous_aliases.items())),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'wrote {len(aliases)} aliases for {len(fluids)} fluids to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
