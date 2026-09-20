import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from property_resolution.runtime_locks import (
    RuntimeLockOwnershipError,
    SQLiteLeaseLock,
    runtime_lock,
)


class RuntimeLockTests(unittest.TestCase):
    def test_acquire_and_release_own_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'locks.sqlite')
            lock = runtime_lock('fixture', 'resource', path=path)
            with lock:
                with sqlite3.connect(path) as connection:
                    row = connection.execute(
                        """
                        SELECT owner_token FROM runtime_locks
                        WHERE namespace = ? AND resource_key = ?
                        """,
                        ('fixture', 'resource'),
                    ).fetchone()
                self.assertEqual(row[0], lock.owner_token)
            with sqlite3.connect(path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM runtime_locks"
                ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_second_owner_times_out_until_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'locks.sqlite')
            with runtime_lock('fixture', 'resource', path=path):
                blocked = SQLiteLeaseLock(
                    path,
                    'fixture',
                    'resource',
                    poll_seconds=0.01,
                    acquire_timeout=0.05,
                )
                with self.assertRaises(TimeoutError):
                    blocked.acquire()
            with SQLiteLeaseLock(
                path,
                'fixture',
                'resource',
                acquire_timeout=0.05,
            ):
                pass

    def test_expired_owner_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'locks.sqlite')
            seed = SQLiteLeaseLock(path, 'fixture', 'resource')
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    INSERT INTO runtime_locks (
                        namespace, resource_key, owner_token, owner_pid,
                        owner_host, acquired_at_utc, renewed_at_utc,
                        expires_at_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        'fixture', 'resource', 'dead-owner', 1, 'test',
                        '2026-01-01T00:00:00+00:00',
                        '2026-01-01T00:00:00+00:00',
                        time.time() - 1.0,
                    ),
                )
            with seed:
                with sqlite3.connect(path) as connection:
                    owner = connection.execute(
                        """
                        SELECT owner_token FROM runtime_locks
                        WHERE namespace = ? AND resource_key = ?
                        """,
                        ('fixture', 'resource'),
                    ).fetchone()[0]
                self.assertEqual(owner, seed.owner_token)

    def test_stale_owner_cannot_release_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'locks.sqlite')
            stale = SQLiteLeaseLock(path, 'fixture', 'resource')
            stale.acquire()
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE runtime_locks
                    SET owner_token = ?
                    WHERE namespace = ? AND resource_key = ?
                    """,
                    ('replacement-owner', 'fixture', 'resource'),
                )
            with self.assertRaises(RuntimeLockOwnershipError):
                stale.__exit__(None, None, None)
            with sqlite3.connect(path) as connection:
                owner = connection.execute(
                    """
                    SELECT owner_token FROM runtime_locks
                    WHERE namespace = ? AND resource_key = ?
                    """,
                    ('fixture', 'resource'),
                ).fetchone()[0]
            self.assertEqual(owner, 'replacement-owner')

    def test_heartbeat_keeps_long_operation_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'locks.sqlite')
            with SQLiteLeaseLock(
                path,
                'fixture',
                'resource',
                lease_seconds=0.15,
                heartbeat_seconds=0.03,
            ):
                time.sleep(0.25)
                contender = SQLiteLeaseLock(
                    path,
                    'fixture',
                    'resource',
                    lease_seconds=0.15,
                    heartbeat_seconds=0.03,
                    poll_seconds=0.01,
                    acquire_timeout=0.05,
                )
                with self.assertRaises(TimeoutError):
                    contender.acquire()


if __name__ == '__main__':
    unittest.main()
