"""Shared SQLite storage for JSON-shaped runtime cache payloads."""

from __future__ import annotations

from contextlib import closing, contextmanager
import json
import re
import sqlite3
import threading
import urllib.parse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from .cache_expiration import (
    cache_expiration_cutoff_utc,
    runtime_cache_row_is_fresh,
    sqlite_cache_ttl_days,
)


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PROPERTY_CACHE_PATH = ROOT / "data" / "runtime" / "property_cache.sqlite"
LEGACY_PROPERTY_CACHE_DIR = ROOT / ".property_cache"
LEGACY_ONLINE_COMPONENT_CACHE_DIR = Path.home() / ".pfd_cache"
RUNTIME_CACHE_SCHEMA_VERSION = 3

_SCHEMA_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[Path] = set()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cache_key_metadata(key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract query provenance without changing the exact historical key."""
    text = str(key)
    provenance_text = ' '.join(
        str(payload.get(name) or '')
        for name in ('source', 'method', '_sources')
    ).lower()
    provider = next(
        (
            name
            for name in ('pubchem', 'nist', 'opsin', 'coolprop')
            if name in text.lower() or name in provenance_text
        ),
        '',
    )
    version_match = re.search(r'(?:^|_)v(\d+)(?:_|$)', text, flags=re.I)
    contract_version = int(version_match.group(1)) if version_match else 1
    known_families = (
        'nist_antoine_rows', 'antoine_missing', 'phase_pubchem', 'phase_nist',
        'pubchem_component_cas', 'pubchem_component', 'pubchem_structure',
        'pubchem_cid', 'formation_nist', 'cp_nist', 'critical', 'hvap',
        'antoine', 'density_pubchem', 'knotts_parachor_fragmentation',
        'viscosity_pubchem',
    )
    family = next((item for item in known_families if text.startswith(item + '_')), 'generic')
    if version_match:
        identifier = text[version_match.end():]
    elif family != 'generic':
        identifier = text[len(family) + 1:]
    else:
        identifier = text
    provenance = {
        name: payload[name]
        for name in ('source', 'method', 'reason', '_sources', '_qualities', '_notes')
        if name in payload
    }
    return {
        'cache_family': family,
        'provider': provider,
        'contract_version': contract_version,
        'identifier_key': identifier,
        'provenance_json': json.dumps(
            provenance,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            default=_json_default,
        ),
    }


def runtime_cache_path_for_legacy_directory(
    directory: Path | str,
    *,
    default_legacy_directory: Path | str,
) -> Path:
    """Use the central database by default and an isolated DB for custom dirs."""
    directory_path = Path(directory).expanduser()
    default_path = Path(default_legacy_directory).expanduser()
    try:
        is_default = directory_path.resolve() == default_path.resolve()
    except OSError:
        is_default = directory_path.absolute() == default_path.absolute()
    if is_default:
        return RUNTIME_PROPERTY_CACHE_PATH
    return directory_path / "property_cache.sqlite"


def runtime_cache_path_for_legacy_file(
    path: Path | str,
    *,
    default_legacy_file: Path | str,
) -> Path:
    """Map a former JSON cache file to the central or a sibling SQLite DB."""
    source = Path(path).expanduser()
    default_source = Path(default_legacy_file).expanduser()
    try:
        is_default = source.resolve() == default_source.resolve()
    except OSError:
        is_default = source.absolute() == default_source.absolute()
    if is_default:
        return RUNTIME_PROPERTY_CACHE_PATH
    return source if source.suffix.lower() in {".sqlite", ".db"} else source.with_suffix(".sqlite")


class SQLiteJSONCache:
    """Namespaced, atomic storage for JSON-compatible runtime cache values."""

    def __init__(self, path: Path | str, namespace: str):
        self.path = Path(path).expanduser()
        self.namespace = str(namespace)
        if not self.namespace:
            raise ValueError("Runtime cache namespace cannot be empty")
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit/roll back each operation, then close its connection."""
        with closing(sqlite3.connect(self.path, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            with connection:
                yield connection

    def _ensure_schema(self) -> None:
        normalized = self.path.absolute()
        if normalized in _INITIALIZED_PATHS and self.path.is_file():
            return
        with _SCHEMA_LOCK:
            if normalized in _INITIALIZED_PATHS and self.path.is_file():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_json_cache (
                        namespace TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        cache_family TEXT NOT NULL DEFAULT 'generic',
                        provider TEXT NOT NULL DEFAULT '',
                        contract_version INTEGER NOT NULL DEFAULT 1,
                        identifier_key TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        is_missing INTEGER NOT NULL DEFAULT 0,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        PRIMARY KEY (namespace, cache_key)
                    )
                    """
                )
                existing_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(runtime_json_cache)")
                }
                migration_columns = {
                    'cache_family': "TEXT NOT NULL DEFAULT 'generic'",
                    'provider': "TEXT NOT NULL DEFAULT ''",
                    'contract_version': "INTEGER NOT NULL DEFAULT 1",
                    'identifier_key': "TEXT NOT NULL DEFAULT ''",
                    'provenance_json': "TEXT NOT NULL DEFAULT '{}'",
                }
                for column_name, column_contract in migration_columns.items():
                    if column_name not in existing_columns:
                        connection.execute(
                            f"ALTER TABLE runtime_json_cache "
                            f"ADD COLUMN {column_name} {column_contract}"
                        )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_runtime_json_cache_namespace_key
                    ON runtime_json_cache(namespace, cache_key)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_runtime_json_cache_source
                    ON runtime_json_cache(provider, cache_family, contract_version, identifier_key)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_cache_migrations (
                        migration_key TEXT PRIMARY KEY,
                        completed_at_utc TEXT NOT NULL,
                        imported_count INTEGER NOT NULL,
                        skipped_count INTEGER NOT NULL,
                        error_count INTEGER NOT NULL
                    )
                    """
                )
                current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if current_version < 3:
                    for row in connection.execute(
                        "SELECT namespace, cache_key, payload_json FROM runtime_json_cache"
                    ).fetchall():
                        try:
                            payload = json.loads(row['payload_json'])
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue
                        if not isinstance(payload, dict):
                            continue
                        metadata = cache_key_metadata(str(row['cache_key']), payload)
                        connection.execute(
                            """
                            UPDATE runtime_json_cache
                            SET cache_family = ?, provider = ?, contract_version = ?,
                                identifier_key = ?, provenance_json = ?
                            WHERE namespace = ? AND cache_key = ?
                            """,
                            (
                                metadata['cache_family'],
                                metadata['provider'],
                                metadata['contract_version'],
                                metadata['identifier_key'],
                                metadata['provenance_json'],
                                row['namespace'],
                                row['cache_key'],
                            ),
                        )
                if current_version < RUNTIME_CACHE_SCHEMA_VERSION:
                    connection.execute(f"PRAGMA user_version = {RUNTIME_CACHE_SCHEMA_VERSION}")
            _INITIALIZED_PATHS.add(normalized)

    @staticmethod
    def _encode_payload(payload: Mapping[str, Any]) -> str:
        if not isinstance(payload, Mapping):
            raise TypeError("Runtime cache payload must be a mapping")
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json, updated_at_utc
                FROM runtime_json_cache
                WHERE namespace = ? AND cache_key = ?
                """,
                (self.namespace, str(key)),
            ).fetchone()
        if row is None:
            return None
        if not runtime_cache_row_is_fresh(
            self.path,
            row['updated_at_utc'],
            namespace=self.namespace,
        ):
            self.delete(key)
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def set(self, key: str, payload: Mapping[str, Any]) -> None:
        encoded = self._encode_payload(payload)
        metadata = cache_key_metadata(str(key), payload)
        timestamp = datetime.now(timezone.utc).isoformat()
        missing = int(bool(payload.get("_missing")))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_json_cache (
                    namespace, cache_key, cache_family, provider,
                    contract_version, identifier_key,
                    payload_json, provenance_json, is_missing,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    cache_family = excluded.cache_family,
                    provider = excluded.provider,
                    contract_version = excluded.contract_version,
                    identifier_key = excluded.identifier_key,
                    payload_json = excluded.payload_json,
                    provenance_json = excluded.provenance_json,
                    is_missing = excluded.is_missing,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    self.namespace,
                    str(key),
                    metadata['cache_family'],
                    metadata['provider'],
                    metadata['contract_version'],
                    metadata['identifier_key'],
                    encoded,
                    metadata['provenance_json'],
                    missing,
                    timestamp,
                    timestamp,
                ),
            )

    def delete(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM runtime_json_cache WHERE namespace = ? AND cache_key = ?",
                (self.namespace, str(key)),
            )

    def items(self, *, prefix: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
        query = (
            "SELECT cache_key, payload_json, updated_at_utc "
            "FROM runtime_json_cache "
            "WHERE namespace = ?"
        )
        parameters: list[Any] = [self.namespace]
        if prefix:
            query += " AND cache_key >= ? AND cache_key < ?"
            parameters.extend([prefix, prefix + "\U0010ffff"])
        query += " ORDER BY cache_key"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        expired = []
        for row in rows:
            if not runtime_cache_row_is_fresh(
                self.path,
                row['updated_at_utc'],
                namespace=self.namespace,
            ):
                expired.append(str(row['cache_key']))
                continue
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                yield str(row["cache_key"]), payload
        if expired:
            with self._connect() as connection:
                connection.executemany(
                    "DELETE FROM runtime_json_cache "
                    "WHERE namespace = ? AND cache_key = ?",
                    [(self.namespace, key) for key in expired],
                )

    def count(self) -> int:
        with self._connect() as connection:
            ttl_days = sqlite_cache_ttl_days(
                self.path,
                namespace=self.namespace,
            )
            if ttl_days is not None:
                cutoff = cache_expiration_cutoff_utc(ttl_days)
                connection.execute(
                    """
                    DELETE FROM runtime_json_cache
                    WHERE namespace = ?
                      AND (
                          datetime(updated_at_utc) IS NULL
                          OR datetime(updated_at_utc) < datetime(?)
                      )
                    """,
                    (self.namespace, cutoff),
                )
            return int(connection.execute(
                "SELECT count(*) FROM runtime_json_cache WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()[0])

    def migrate_json_directory(
        self,
        directory: Path | str,
        *,
        migration_name: str,
        patterns: Iterable[str] = ("*.json",),
    ) -> dict[str, int]:
        """Idempotently import legacy URL-encoded one-payload-per-file caches."""
        source_directory = Path(directory).expanduser()
        migration_key = (
            f"json-directory-v1:{self.namespace}:{migration_name}:"
            f"{source_directory.absolute()}"
        )
        with self._connect() as connection:
            completed = connection.execute(
                "SELECT 1 FROM runtime_cache_migrations WHERE migration_key = ?",
                (migration_key,),
            ).fetchone()
        if completed is not None:
            return {"imported": 0, "skipped": 0, "errors": 0, "already_completed": 1}

        paths: set[Path] = set()
        if source_directory.is_dir():
            for pattern in patterns:
                paths.update(source_directory.glob(pattern))
        imported = skipped = errors = 0
        for path in sorted(paths):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict):
                    raise TypeError("legacy cache payload is not a mapping")
                key = urllib.parse.unquote(path.stem)
                if self.get(key) is not None:
                    skipped += 1
                    continue
                self.set(key, payload)
                imported += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                errors += 1

        if errors == 0:
            timestamp = datetime.now(timezone.utc).isoformat()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO runtime_cache_migrations (
                        migration_key, completed_at_utc,
                        imported_count, skipped_count, error_count
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (migration_key, timestamp, imported, skipped, errors),
                )
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "already_completed": 0,
        }
