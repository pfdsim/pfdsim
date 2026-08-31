"""Central freshness policy for persistent runtime SQLite caches."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


SELECTED_PROPERTY_CACHE_TTL_DAYS = 30
ONLINE_SOURCE_CACHE_TTL_DAYS = 90
SMILES_CACHE_FILENAME = 'smiles_cache.sqlite'
ONLINE_SOURCE_CACHE_FILENAME = 'property_cache.sqlite'
DERIVED_LOCAL_PROPERTY_NAMESPACES = frozenset({
    'liquid_volume_zra_v1',
    'ideal_gas_cp_derived_v1',
    'liquid_cp_derived_v1',
})
SOURCE_DATABASE_FILENAMES = frozenset({
    'effective_criticals.sqlite',
    'henry_constants.sqlite',
    'ideal_gas_heat_capacity.sqlite',
    'liquid_heat_capacity.sqlite',
})


def sqlite_cache_ttl_days(
    path: Path | str,
    *,
    namespace: str = '',
) -> Optional[int]:
    """Return the TTL for a runtime cache, or ``None`` when exempt.

    Source databases such as effective criticals and Henry constants never
    call this helper.  The only exempt runtime cache is the stable molecular-
    structure cache.
    """
    filename = Path(path).name.lower()
    if filename in SOURCE_DATABASE_FILENAMES:
        return None
    if filename == SMILES_CACHE_FILENAME:
        return None
    if (
        str(namespace)
        and str(namespace) not in DERIVED_LOCAL_PROPERTY_NAMESPACES
    ):
        return ONLINE_SOURCE_CACHE_TTL_DAYS
    return SELECTED_PROPERTY_CACHE_TTL_DAYS


def parse_cache_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 cache timestamp as an aware UTC datetime."""
    text = str(value or '').strip()
    if not text:
        return None
    if text.endswith(('Z', 'z')):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def cache_timestamp_is_fresh(
    value: Any,
    *,
    ttl_days: Optional[int],
    now: Optional[datetime] = None,
) -> bool:
    """Return whether a stored timestamp is inside its cache TTL."""
    if ttl_days is None:
        return True
    timestamp = parse_cache_timestamp(value)
    if timestamp is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    else:
        reference = reference.astimezone(timezone.utc)
    age = reference - timestamp
    return timedelta(0) <= age <= timedelta(days=int(ttl_days))


def cache_expiration_cutoff_utc(
    ttl_days: int,
    *,
    now: Optional[datetime] = None,
) -> str:
    """Return an ISO UTC cutoff suitable for a bounded SQLite purge."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    else:
        reference = reference.astimezone(timezone.utc)
    return (reference - timedelta(days=int(ttl_days))).isoformat()


def runtime_cache_row_is_fresh(
    path: Path | str,
    timestamp: Any,
    *,
    namespace: str = '',
    now: Optional[datetime] = None,
) -> bool:
    """Apply the central SQLite cache policy to one stored row."""
    return cache_timestamp_is_fresh(
        timestamp,
        ttl_days=sqlite_cache_ttl_days(path, namespace=namespace),
        now=now,
    )
