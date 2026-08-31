"""Canonical recycle-solver method names and numerical controls."""

from __future__ import annotations

import math
from collections.abc import Mapping


RECYCLE_METHOD_ALIASES = {
    'WEGSTEIN': 'WEGSTEIN',
    'WEGSTEIN_ACCELERATION': 'WEGSTEIN',
    'DIRECT': 'DIRECT',
    'DIRECT_SUBSTITUTION': 'DIRECT',
    'SUBSTITUTION': 'DIRECT',
    'BROYDEN': 'BROYDEN',
    'BROYDEN1': 'BROYDEN',
    'NEWTON_BROYDEN': 'BROYDEN',
}


RECYCLE_METHOD_DEFAULTS = {
    'WEGSTEIN': {
        'max_acceleration': -5.0,
        'stagnation_iterations': 8,
        'fallback_damping': 1.0,
    },
    'BROYDEN': {
        'stagnation_iterations': 8,
        'divergence_factor': 10.0,
        'fallback_damping': 1.0,
    },
    'DIRECT': {
        'damping': 1.0,
    },
}


def normalize_recycle_method(value: object) -> str:
    """Return one canonical recycle method or raise ``ValueError``."""
    token = str(value or '').strip().upper().replace('-', '_')
    method = RECYCLE_METHOD_ALIASES.get(token)
    if method is None:
        supported = ', '.join(RECYCLE_METHOD_DEFAULTS)
        raise ValueError(
            f"Unknown recycle method {value!r}; use {supported}."
        )
    return method


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_damping(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be greater than 0 and at most 1")
    return result


def _positive_integer(value: object, label: str) -> int:
    result = _finite_float(value, label)
    integer = int(result)
    if result != integer or integer < 1:
        raise ValueError(f"{label} must be a positive integer")
    return integer


def normalize_recycle_options(
    method: object,
    options: Mapping[str, object] | None,
) -> dict[str, float | int]:
    """Validate configured options for one canonical recycle method."""
    canonical_method = normalize_recycle_method(method)
    supplied = dict(options or {})
    supported = RECYCLE_METHOD_DEFAULTS[canonical_method]
    unknown = sorted(set(supplied) - set(supported))
    if unknown:
        names = ', '.join(sorted(supported))
        raise ValueError(
            f"Recycle method {canonical_method} does not support option "
            f"{unknown[0]!r}; supported options are {names}."
        )

    normalized: dict[str, float | int] = {}
    for name, value in supplied.items():
        label = f"Recycle option {name}"
        if name == 'max_acceleration':
            result = _finite_float(value, label)
            if not -100.0 <= result <= 0.0:
                raise ValueError(
                    f"{label} must be between -100 and 0"
                )
            normalized[name] = result
        elif name == 'stagnation_iterations':
            normalized[name] = _positive_integer(value, label)
        elif name in {'fallback_damping', 'damping'}:
            normalized[name] = _positive_damping(value, label)
        elif name == 'divergence_factor':
            result = _finite_float(value, label)
            if result <= 1.0:
                raise ValueError(f"{label} must be greater than 1")
            normalized[name] = result
        else:  # Defensive: the supported table and coercion must stay aligned.
            raise ValueError(f"Unsupported recycle option {name!r}")
    return normalized


def resolved_recycle_options(
    method: object,
    options: Mapping[str, object] | None,
) -> dict[str, float | int]:
    """Return validated method options overlaid on stable defaults."""
    canonical_method = normalize_recycle_method(method)
    resolved = dict(RECYCLE_METHOD_DEFAULTS[canonical_method])
    resolved.update(normalize_recycle_options(canonical_method, options))
    return resolved

