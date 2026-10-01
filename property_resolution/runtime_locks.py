"""Cross-process runtime coordination backed by renewable SQLite leases."""

from __future__ import annotations

from contextlib import closing, contextmanager
import os
import socket
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


ROOT = Path(__file__).resolve().parent.parent
RUNTIME_LOCKS_PATH = ROOT / "data" / "runtime" / "locks.sqlite"


class RuntimeLockOwnershipError(RuntimeError):
    """Raised when a lease owner loses its lock before releasing it."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sqlite_contention(error: sqlite3.OperationalError) -> bool:
    return getattr(error, "sqlite_errorcode", None) in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }


class SQLiteLeaseLock:
    """One renewable, crash-recoverable lock in the shared runtime registry."""

    def __init__(
        self,
        path: Path | str,
        namespace: str,
        resource_key: str,
        *,
        lease_seconds: float = 300.0,
        heartbeat_seconds: Optional[float] = None,
        poll_seconds: float = 0.25,
        acquire_timeout: Optional[float] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.namespace = str(namespace).strip()
        self.resource_key = str(resource_key).strip()
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_seconds = float(
            heartbeat_seconds
            if heartbeat_seconds is not None
            else min(60.0, self.lease_seconds / 3.0)
        )
        self.poll_seconds = float(poll_seconds)
        self.acquire_timeout = (
            None if acquire_timeout is None else float(acquire_timeout)
        )
        if not self.namespace or not self.resource_key:
            raise ValueError("Runtime lock namespace and resource key are required")
        if self.lease_seconds <= 0.0 or self.heartbeat_seconds <= 0.0:
            raise ValueError("Runtime lock lease and heartbeat must be positive")
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("Runtime lock heartbeat must be shorter than its lease")
        if self.poll_seconds <= 0.0:
            raise ValueError("Runtime lock polling interval must be positive")
        if self.acquire_timeout is not None and self.acquire_timeout < 0.0:
            raise ValueError("Runtime lock acquisition timeout cannot be negative")

        self.owner_token = uuid.uuid4().hex
        self._acquired = False
        self._stop_heartbeat = threading.Event()
        self._lost_ownership = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Close lease connections after committing or rolling back."""
        with closing(sqlite3.connect(self.path, timeout=1.0)) as connection:
            connection.execute("PRAGMA busy_timeout = 1000")
            with connection:
                yield connection

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_locks (
                    namespace TEXT NOT NULL,
                    resource_key TEXT NOT NULL,
                    owner_token TEXT NOT NULL,
                    owner_pid INTEGER NOT NULL,
                    owner_host TEXT NOT NULL,
                    acquired_at_utc TEXT NOT NULL,
                    renewed_at_utc TEXT NOT NULL,
                    expires_at_epoch REAL NOT NULL,
                    PRIMARY KEY (namespace, resource_key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_locks_expiration
                ON runtime_locks(expires_at_epoch)
                """
            )

    def _try_acquire(self) -> bool:
        now_epoch = time.time()
        now_utc = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runtime_locks (
                    namespace, resource_key, owner_token, owner_pid, owner_host,
                    acquired_at_utc, renewed_at_utc, expires_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, resource_key) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    owner_pid = excluded.owner_pid,
                    owner_host = excluded.owner_host,
                    acquired_at_utc = excluded.acquired_at_utc,
                    renewed_at_utc = excluded.renewed_at_utc,
                    expires_at_epoch = excluded.expires_at_epoch
                WHERE runtime_locks.expires_at_epoch <= ?
                """,
                (
                    self.namespace,
                    self.resource_key,
                    self.owner_token,
                    os.getpid(),
                    socket.gethostname(),
                    now_utc,
                    now_utc,
                    now_epoch + self.lease_seconds,
                    now_epoch,
                ),
            )
            return cursor.rowcount == 1

    def acquire(self) -> "SQLiteLeaseLock":
        if self._acquired:
            raise RuntimeError("Runtime lock is already acquired")
        deadline = (
            None
            if self.acquire_timeout is None
            else time.monotonic() + self.acquire_timeout
        )
        while True:
            try:
                acquired = self._try_acquire()
            except sqlite3.OperationalError as error:
                if not _is_sqlite_contention(error):
                    raise
                acquired = False
            if acquired:
                self._acquired = True
                self._stop_heartbeat.clear()
                self._lost_ownership.clear()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name=f"runtime-lock-{self.namespace}",
                    daemon=True,
                )
                self._heartbeat_thread.start()
                return self
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out acquiring runtime lock "
                        f"{self.namespace}:{self.resource_key}"
                    )
                time.sleep(min(self.poll_seconds, remaining))
            else:
                time.sleep(self.poll_seconds)

    def _renew(self) -> bool:
        now_epoch = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runtime_locks
                SET renewed_at_utc = ?, expires_at_epoch = ?
                WHERE namespace = ? AND resource_key = ? AND owner_token = ?
                """,
                (
                    _utc_now(),
                    now_epoch + self.lease_seconds,
                    self.namespace,
                    self.resource_key,
                    self.owner_token,
                ),
            )
            return cursor.rowcount == 1

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self.heartbeat_seconds):
            try:
                if not self._renew():
                    self._lost_ownership.set()
                    return
            except sqlite3.OperationalError as error:
                if not _is_sqlite_contention(error):
                    self._lost_ownership.set()
                    return

    def release(self) -> bool:
        if not self._acquired:
            return False
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
            self._heartbeat_thread = None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM runtime_locks
                WHERE namespace = ? AND resource_key = ? AND owner_token = ?
                """,
                (self.namespace, self.resource_key, self.owner_token),
            )
        self._acquired = False
        released = cursor.rowcount == 1 and not self._lost_ownership.is_set()
        return released

    def __enter__(self) -> "SQLiteLeaseLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            released = self.release()
        except Exception:
            if exc_type is None:
                raise
            return False
        if not released and exc_type is None:
            raise RuntimeLockOwnershipError(
                f"Lost runtime lock {self.namespace}:{self.resource_key}"
            )
        return False


def runtime_lock(
    namespace: str,
    resource_key: str,
    *,
    path: Path | str = RUNTIME_LOCKS_PATH,
    lease_seconds: float = 300.0,
    heartbeat_seconds: Optional[float] = None,
    poll_seconds: float = 0.25,
    acquire_timeout: Optional[float] = None,
) -> SQLiteLeaseLock:
    """Return a context-managed lease in the shared runtime lock registry."""
    return SQLiteLeaseLock(
        path,
        namespace,
        resource_key,
        lease_seconds=lease_seconds,
        heartbeat_seconds=heartbeat_seconds,
        poll_seconds=poll_seconds,
        acquire_timeout=acquire_timeout,
    )
