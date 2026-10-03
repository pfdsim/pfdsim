"""Private account storage, versioned flowsheets, and daily compute accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time

from werkzeug.security import check_password_hash, generate_password_hash

GUEST_CPU_SECONDS = 300
ACCOUNT_CPU_SECONDS = 900


class StorageConflict(ValueError):
    pass


class AccessLimit(ValueError):
    pass


class WebStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "web.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE,
                    display_name TEXT, password_hash TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS flowsheets (
                    id TEXT PRIMARY KEY, owner TEXT, version INTEGER,
                    updated REAL, document TEXT);
                CREATE TABLE IF NOT EXISTS usage (
                    principal TEXT, day TEXT, cpu_seconds REAL,
                    PRIMARY KEY (principal,day));
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    principal TEXT, timestamp REAL);
                CREATE INDEX IF NOT EXISTS auth_attempts_principal
                    ON auth_attempts(principal,timestamp);
            """)

    @contextmanager
    def connect(self):
        database = sqlite3.connect(self.path, timeout=10)
        try:
            with database:
                yield database
        finally:
            database.close()

    @staticmethod
    def day():
        return datetime.now(timezone.utc).date().isoformat()

    def quota(self, principal, limit):
        day = self.day()
        with self.connect() as db:
            row = db.execute(
                "SELECT cpu_seconds FROM usage WHERE principal=? AND day=?",
                (principal, day),
            ).fetchone()
        used = row[0] if row else 0.0
        return {
            "day": day,
            "limit_seconds": limit,
            "used_seconds": used,
            "remaining_seconds": max(0.0, limit - used),
            "reset_timezone": "UTC",
        }

    def charge(self, principal, limit, seconds, *, day=None):
        """Serialize debits across both computation workers, never HTTP workers."""
        day = day or self.day()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO usage VALUES (?,?,?) ON CONFLICT(principal,day) DO UPDATE SET cpu_seconds=cpu_seconds+excluded.cpu_seconds",
                (principal, day, max(0.0, seconds)),
            )
            used = db.execute(
                "SELECT cpu_seconds FROM usage WHERE principal=? AND day=?",
                (principal, day),
            ).fetchone()[0]
        return used >= limit

    def auth_throttle(self, principal):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM auth_attempts WHERE timestamp<?", (now - 300,))
            count = db.execute(
                "SELECT count(*) FROM auth_attempts WHERE principal=?", (principal,)
            ).fetchone()[0]
            if count >= 10:
                raise AccessLimit(
                    "Too many account attempts. Try again in five minutes."
                )
            db.execute("INSERT INTO auth_attempts VALUES (?,?)", (principal, now))

    def register(self, username, password, principal):
        self.auth_throttle(principal)
        if not isinstance(username, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{3,32}", username
        ):
            raise ValueError(
                "Use a username of 3–32 letters, numbers, dots, underscores, or hyphens."
            )
        if not isinstance(password, str) or not 10 <= len(password) <= 256:
            raise ValueError("Use a password of 10–256 characters.")
        identifier = secrets.token_urlsafe(24)
        password_hash = generate_password_hash(password, method="scrypt")
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO users VALUES (?,?,?,?,?)",
                    (
                        identifier,
                        username.casefold(),
                        username,
                        password_hash,
                        time.time(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StorageConflict("That username is already in use.") from error
        return {"id": identifier, "username": username}

    def login(self, username, password, principal):
        self.auth_throttle(principal)
        if (
            not isinstance(username, str)
            or not isinstance(password, str)
            or len(password) > 256
        ):
            raise ValueError("Invalid username or password.")
        with self.connect() as db:
            row = db.execute(
                "SELECT id,display_name,password_hash FROM users WHERE username=?",
                (username.casefold(),),
            ).fetchone()
        if row is None or not check_password_hash(row[2], password):
            raise ValueError("Invalid username or password.")
        return {"id": row[0], "username": row[1]}

    def user(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT id,display_name FROM users WHERE id=?", (identifier,)
            ).fetchone()
        return {"id": row[0], "username": row[1]} if row else None

    def flowsheets(self, owner):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,version,updated,document FROM flowsheets WHERE owner=? ORDER BY updated DESC",
                (owner,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "version": row[1],
                "updated": row[2],
                "document": json.loads(row[3]),
            }
            for row in rows
        ]

    def save_flowsheet(self, owner, identifier, expected_version, document):
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[a-zA-Z0-9_-]{8,80}", identifier
        ):
            raise ValueError("Invalid laboratory identifier.")
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise ValueError("A nonnegative saved version is required.")
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("text"), str)
            or not isinstance(document.get("pfd"), dict)
        ):
            raise ValueError(
                "Provide the applied PFD object and its source or text draft."
            )
        if not isinstance(document.get("pending", False), bool) or not isinstance(
            document.get("filename", "process.pfd"), str
        ):
            raise ValueError("Invalid laboratory document.")
        # Only the editor document is saved; account and worker metadata are
        # server-owned. Invalid *text drafts* are deliberately accepted.
        document = {
            key: document.get(key)
            for key in ("pfd", "text", "pending", "filename", "job", "lastJob", "layout")
        }
        encoded = json.dumps(document, allow_nan=False)
        now = time.time() * 1000
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT owner,version FROM flowsheets WHERE id=?", (identifier,)
            ).fetchone()
            if current is not None and current[0] != owner:
                raise StorageConflict(
                    "This laboratory belongs to another account. Save a new copy."
                )
            if (current[1] if current else 0) != expected_version:
                raise StorageConflict(
                    "Another session saved this laboratory. Your local draft is safe; open the saved laboratory list to review both versions."
                )
            version = expected_version + 1
            db.execute(
                "INSERT INTO flowsheets VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET version=excluded.version,updated=excluded.updated,document=excluded.document",
                (identifier, owner, version, now, encoded),
            )
        return {"id": identifier, "version": version, "updated": now}
