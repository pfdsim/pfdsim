"""Private account storage, versioned flowsheets, and daily compute accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time

from werkzeug.security import check_password_hash, generate_password_hash

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .activity_fit_store import ActivityFitStore
else:
    from activity_fit_store import ActivityFitStore

GUEST_CPU_SECONDS = 300
ACCOUNT_CPU_SECONDS = 900
ACCOUNT_CPU_GRACE_SECONDS = 15


class StorageConflict(ValueError):
    pass


class AccessLimit(ValueError):
    pass


class WebStore:
    def __init__(self, directory, *, activity_fits_path=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "web.sqlite"
        self.activity_fits_path = Path(activity_fits_path or self.directory / "user_activity_fits.sqlite")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE,
                    display_name TEXT, password_hash TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS flowsheets (
                    id TEXT PRIMARY KEY, owner TEXT, version INTEGER,
                    updated REAL, document TEXT);
                CREATE TABLE IF NOT EXISTS fit_sessions (
                    id TEXT PRIMARY KEY, owner TEXT, version INTEGER,
                    updated REAL, document TEXT);
                CREATE TABLE IF NOT EXISTS usage (
                    principal TEXT, day TEXT, cpu_seconds REAL,
                    PRIMARY KEY (principal,day));
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    principal TEXT, timestamp REAL);
                CREATE TABLE IF NOT EXISTS account_roles (
                    user_id TEXT PRIMARY KEY, role TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS auth_attempts_principal
                    ON auth_attempts(principal,timestamp);
            """)
        # A generated local token is never sent through HTTP. Claiming root
        # consumes the credential by committing its role in the same transaction.
        with self.connect() as db:
            root_exists = db.execute("SELECT 1 FROM users WHERE username='root'").fetchone()
        self.root_token_path = self.directory / "root-setup-token"
        if not root_exists and not os.environ.get("PFDSIM_ROOT_SETUP_TOKEN"):
            try:
                descriptor = os.open(self.root_token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(descriptor,"w") as file:
                    file.write(secrets.token_urlsafe(32)+"\n")

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

    def submit_fit(self, owner, job_id, source, result):
        """Persist a sourced review artifact independently of job expiration."""
        return ActivityFitStore(self.activity_fits_path).submit(owner,job_id,source,result)

    def fit_submissions(self, owner):
        return ActivityFitStore(self.activity_fits_path).list(owner)

    @staticmethod
    def _cpu_allowance(db, principal, limit):
        """Resolve account limits and running-job grace from the stored identity."""
        if principal.startswith("user:"):
            user = db.execute(
                "SELECT username FROM users WHERE id=?",
                (principal.removeprefix("user:"),),
            ).fetchone()
            if user:
                if user[0] == "root":
                    return None, 0
                return limit, ACCOUNT_CPU_GRACE_SECONDS
        return limit, 0

    def quota(self, principal, limit):
        day = self.day()
        with self.connect() as db:
            limit, grace = self._cpu_allowance(db, principal, limit)
            row = db.execute(
                "SELECT cpu_seconds FROM usage WHERE principal=? AND day=?",
                (principal, day),
            ).fetchone()
        used = row[0] if row else 0.0
        return {
            "day": day,
            "limit_seconds": limit,
            "used_seconds": used,
            "remaining_seconds": None if limit is None else max(0.0, limit - used),
            "grace_seconds": grace,
            "reset_timezone": "UTC",
        }

    def charge(self, principal, limit, seconds, *, day=None, allow_grace=False):
        """Serialize debits across both computation workers, never HTTP workers."""
        day = day or self.day()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            limit, grace = self._cpu_allowance(db, principal, limit)
            if limit is not None and allow_grace:
                limit += grace
            db.execute(
                "INSERT INTO usage VALUES (?,?,?) ON CONFLICT(principal,day) DO UPDATE SET cpu_seconds=cpu_seconds+excluded.cpu_seconds",
                (principal, day, max(0.0, seconds)),
            )
            used = db.execute(
                "SELECT cpu_seconds FROM usage WHERE principal=? AND day=?",
                (principal, day),
            ).fetchone()[0]
        return limit is not None and used >= limit

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

    def register(self, username, password, principal, *, setup_token=None):
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
                db.execute("BEGIN IMMEDIATE")
                if username.casefold() == "root":
                    if db.execute("SELECT 1 FROM users WHERE username='root'").fetchone():
                        raise StorageConflict("The root account has already been claimed.")
                    expected = os.environ.get("PFDSIM_ROOT_SETUP_TOKEN") or self.root_token_path.read_text().strip()
                    if len(expected)<32 or not isinstance(setup_token,str) or not secrets.compare_digest(setup_token,expected):
                        raise ValueError("The root account requires the one-time setup token from the server’s root-setup-token file.")
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
                if username.casefold() == "root":
                    db.execute("INSERT INTO account_roles VALUES (?,?)", (identifier,"admin"))
        except sqlite3.IntegrityError as error:
            raise StorageConflict("That username is already in use.") from error
        return self.user(identifier)

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
        return self.user(row[0])

    def user(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT id,display_name FROM users WHERE id=?", (identifier,)
            ).fetchone()
            role = db.execute("SELECT role FROM account_roles WHERE user_id=?", (identifier,)).fetchone()
        return {"id": row[0], "username": row[1], "is_admin": bool(role and role[0] == "admin")} if row else None

    def flowsheets(self, owner):
        return self.saved_documents("flowsheets",owner)

    def saved_documents(self, table, owner):
        if table not in ("flowsheets","fit_sessions"):
            raise ValueError("Unknown saved-document collection.")
        with self.connect() as db:
            rows = db.execute(
                f"SELECT id,version,updated,document FROM {table} WHERE owner=? ORDER BY updated DESC",
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
        return self.save_document("flowsheets",owner,identifier,expected_version,document)

    def save_document(self, table, owner, identifier, expected_version, document):
        if table not in ("flowsheets","fit_sessions"):
            raise ValueError("Unknown saved-document collection.")
        if not isinstance(identifier,str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}",identifier):
            raise ValueError("Invalid saved-document identifier.")
        if isinstance(expected_version,bool) or not isinstance(expected_version,int) or expected_version<0:
            raise ValueError("A nonnegative saved version is required.")
        encoded = json.dumps(document, allow_nan=False)
        now = time.time() * 1000
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                f"SELECT owner,version FROM {table} WHERE id=?", (identifier,)
            ).fetchone()
            if current is not None and current[0] != owner:
                raise StorageConflict(
                    "This saved document belongs to another account. Save a new copy."
                )
            if (current[1] if current else 0) != expected_version:
                raise StorageConflict(
                    "Another session saved this document. Your local draft is safe; reload the saved list or save a new copy."
                )
            version = expected_version + 1
            db.execute(
                f"INSERT INTO {table} VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET version=excluded.version,updated=excluded.updated,document=excluded.document",
                (identifier, owner, version, now, encoded),
            )
        return {"id": identifier, "version": version, "updated": now}

    def save_fit_session(self,owner,identifier,expected_version,document):
        if not isinstance(document,dict) or document.get("type")!="pfdsim_fit_session" or document.get("schema_version")!=1:
            raise ValueError("Provide a PFDSim fitting-session document.")
        if not isinstance(document.get("name"),str) or not document["name"].strip() or len(document["name"])>120:
            raise ValueError("Give the fitting session a name of 1–120 characters.")
        state=document.get("state")
        if not isinstance(state,dict) or not isinstance(state.get("controls"),dict) or not isinstance(state.get("observations"),list):
            raise ValueError("A fitting session must contain its controls and observation data.")
        document={"type":"pfdsim_fit_session","schema_version":1,"name":document["name"].strip(),"state":state}
        return self.save_document("fit_sessions",owner,identifier,expected_version,document)
