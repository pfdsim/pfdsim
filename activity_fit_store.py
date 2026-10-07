"""Sourced activity-fit submissions and auditable shared-builder publication."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time


DEFAULT_ACTIVITY_FITS_PATH = (
    Path(__file__).resolve().parent
    / "data/source/activity_fitting/user_activity_fits.sqlite"
)


class ActivityFitStore:
    def __init__(self, path=None):
        self.path = Path(
            path
            or os.environ.get("PFDSIM_ACTIVITY_FITS_PATH", DEFAULT_ACTIVITY_FITS_PATH)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS fits (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, job_id TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, status TEXT NOT NULL,
                    source TEXT NOT NULL, result TEXT NOT NULL, model TEXT NOT NULL,
                    cas1 TEXT, cas2 TEXT, reviewed_by TEXT, review_notes TEXT,
                    published REAL, version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(owner,job_id));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, fit_id TEXT NOT NULL,
                    timestamp REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
                    details TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS fits_model_status ON fits(model,status);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def submit(self, owner, job_id, source, result):
        if not isinstance(source, dict) or source.keys() - {
            "citation",
            "url",
            "doi",
            "notes",
        }:
            raise ValueError(
                "Source must contain citation and optional url, doi and notes."
            )
        if (
            not isinstance(source.get("citation"), str)
            or not source["citation"].strip()
        ):
            raise ValueError("A source citation is required for general inclusion.")
        if any(
            not isinstance(value, str) or len(value) > 10000
            for value in source.values()
        ):
            raise ValueError("Source fields must be text of at most 10000 characters.")
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            raise ValueError(
                "Provide a completed fit report with a boolean success assessment."
            )
        source, result = deepcopy(source), deepcopy(result)
        encoded = json.dumps(result, allow_nan=False, sort_keys=True)
        cas = result.get("component_cas") or [None, None]
        if not isinstance(cas, list) or len(cas) != 2:
            raise ValueError("Fit component_cas must contain two identities.")
        identifier, now = secrets.token_urlsafe(24), time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id,status FROM fits WHERE owner=? AND job_id=?", (owner, job_id)
            ).fetchone()
            if existing:
                return dict(existing)
            db.execute(
                "INSERT INTO fits(id,owner,job_id,created,updated,status,source,result,model,cas1,cas2) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    owner,
                    job_id,
                    now,
                    now,
                    "pending_review",
                    json.dumps(source),
                    encoded,
                    result["model"],
                    *cas,
                ),
            )
            db.execute(
                "INSERT INTO events(fit_id,timestamp,actor,action,details) VALUES (?,?,?,?,?)",
                (
                    identifier,
                    now,
                    owner,
                    "submitted",
                    json.dumps(
                        {
                            "source": source,
                            "result_sha256": hashlib.sha256(
                                encoded.encode()
                            ).hexdigest(),
                            "origin": result.get("submission_origin", "web_job"),
                        }
                    ),
                ),
            )
        return {"id": identifier, "status": "pending_review"}

    def list(self, owner=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,owner,created,updated,status,source,model,cas1,cas2,reviewed_by,review_notes,published,version,result FROM fits"
                + (" WHERE owner=?" if owner else "")
                + " ORDER BY created DESC",
                (owner,) if owner else (),
            ).fetchall()
        summaries = []
        for row in rows:
            item = dict(row)
            result = json.loads(item.pop("result"))
            summaries.append(
                {
                    **item,
                    "source": json.loads(row["source"]),
                    "component_names": result.get("component_names", []),
                    "method": result.get("method", item["model"]),
                    "objectives": list(result.get("objectives", {})),
                }
            )
        return summaries

    def claim_owner(self, previous_owner, owner):
        """Transfer a guest's review artifacts without discarding duplicate history."""
        if previous_owner == owner:
            return
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT id,job_id FROM fits WHERE owner=?", (previous_owner,)
            ).fetchall()
            for row in rows:
                job_id = row["job_id"]
                if db.execute(
                    "SELECT 1 FROM fits WHERE owner=? AND job_id=?", (owner, job_id)
                ).fetchone():
                    # Identical external reports share a content-based job ID,
                    # but their sources, decisions and publication histories
                    # can differ. Preserve both submission IDs and histories.
                    job_id = "claimed:" + row["id"] + ":" + job_id
                now = time.time()
                db.execute(
                    "UPDATE fits SET owner=?,job_id=?,updated=?,version=version+1 WHERE id=?",
                    (owner, job_id, now, row["id"]),
                )
                db.execute(
                    "INSERT INTO events(fit_id,timestamp,actor,action,details) VALUES (?,?,?,?,?)",
                    (row["id"], now, owner, "owner_claimed", json.dumps({
                        "previous_owner": previous_owner,
                        "previous_job_id": row["job_id"],
                    })),
                )

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM fits WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Fit submission not found.")
            events = db.execute(
                "SELECT timestamp,actor,action,details FROM events WHERE fit_id=? ORDER BY id",
                (identifier,),
            ).fetchall()
        return {
            **dict(row),
            "source": json.loads(row["source"]),
            "result": json.loads(row["result"]),
            "events": [
                {**dict(event), "details": json.loads(event["details"])}
                for event in events
            ],
        }

    def review(self, identifier, actor, action, notes, *, expected_version):
        if action not in ("approve", "reject", "revoke"):
            raise ValueError("Review action must be approve, reject or revoke.")
        if notes is not None and not isinstance(notes, str):
            raise ValueError("Review notes must be text.")
        notes = (notes or "").strip() or {
            "approve": "Approved by administrator.",
            "reject": "Rejected by administrator.",
            "revoke": "Revoked by administrator.",
        }[action]
        target = {"approve": "approved", "reject": "rejected", "revoke": "revoked"}[
            action
        ]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,version FROM fits WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("Fit submission not found.")
            if row["version"] != expected_version:
                raise ValueError("This submission changed; reload it before reviewing.")
            if row["status"] in ("published", "publishing"):
                raise ValueError(
                    "Published fits must be withdrawn through the builder publication workflow."
                )
            db.execute(
                "UPDATE fits SET status=?,updated=?,reviewed_by=?,review_notes=?,version=version+1 WHERE id=?",
                (target, time.time(), actor, notes, identifier),
            )
            db.execute(
                "INSERT INTO events(fit_id,timestamp,actor,action,details) VALUES (?,?,?,?,?)",
                (
                    identifier,
                    time.time(),
                    actor,
                    action,
                    json.dumps({"notes": notes, "previous_status": row["status"]}),
                ),
            )
        return self.get(identifier)

    def publication_state(self, identifier, actor, action, details):
        """Builder-owned state transition, recorded alongside immutable artifacts."""
        targets = {
            "publishing": "publishing",
            "published": "published",
            "publish_failed": "approved",
            "withdrawn": "revoked",
        }
        if action not in targets:
            raise ValueError("Unsupported publication event.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,model,cas1,cas2 FROM fits WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("Fit submission not found.")
            if action == "publishing" and row["status"] not in (
                "approved",
                "published",
                "publishing",
            ):
                raise ValueError("Approve the submission before publication.")
            if action == "publishing":
                previous_status = row["status"]
                if previous_status == "publishing":
                    previous_event = db.execute(
                        "SELECT details FROM events WHERE fit_id=? AND action='publishing' ORDER BY id DESC LIMIT 1",
                        (identifier,),
                    ).fetchone()
                    previous_status = (
                        json.loads(previous_event["details"]).get(
                            "previous_status", "approved"
                        )
                        if previous_event
                        else "approved"
                    )
                details = {**details, "previous_status": previous_status}
            if action == "publish_failed":
                previous = db.execute(
                    "SELECT details FROM events WHERE fit_id=? AND action='publishing' ORDER BY id DESC LIMIT 1",
                    (identifier,),
                ).fetchone()
                if (
                    previous
                    and json.loads(previous["details"]).get("previous_status")
                    == "published"
                ):
                    targets[action] = "published"
            if action == "published":
                # One authoritative active user fit per model/pair; historical
                # records and their coefficients remain fully recoverable.
                replaced = db.execute(
                    "SELECT id FROM fits WHERE model=? AND ((cas1=? AND cas2=?) OR (cas1=? AND cas2=?)) AND id<>? AND status='published'",
                    (
                        row["model"],
                        row["cas1"],
                        row["cas2"],
                        row["cas2"],
                        row["cas1"],
                        identifier,
                    ),
                ).fetchall()
                for previous in replaced:
                    db.execute(
                        "UPDATE fits SET status='superseded',updated=?,version=version+1 WHERE id=?",
                        (time.time(), previous["id"]),
                    )
                    db.execute(
                        "INSERT INTO events(fit_id,timestamp,actor,action,details) VALUES (?,?,?,?,?)",
                        (
                            previous["id"],
                            time.time(),
                            actor,
                            "superseded",
                            json.dumps({"replacement": identifier}),
                        ),
                    )
            db.execute(
                "UPDATE fits SET status=?,updated=?,published=CASE WHEN ?='published' THEN ? ELSE published END,version=version+1 WHERE id=?",
                (targets[action], time.time(), action, time.time(), identifier),
            )
            db.execute(
                "INSERT INTO events(fit_id,timestamp,actor,action,details) VALUES (?,?,?,?,?)",
                (
                    identifier,
                    time.time(),
                    actor,
                    action,
                    json.dumps(details, allow_nan=False),
                ),
            )

    def runtime_records(self, model, *, candidate=None, exclude=None):
        """The shared builder consumes published fits plus an explicit candidate."""
        entries = []
        for item in self.list():
            if item["model"] != model or item["id"] == exclude:
                continue
            if item["status"] == "published":
                entries.append(item)
            elif item["status"] == "publishing":
                full = self.get(item["id"])
                latest = next(
                    (
                        event
                        for event in reversed(full["events"])
                        if event["action"] == "publishing"
                    ),
                    None,
                )
                if latest and latest["details"].get("previous_status") == "published":
                    entries.append(full)
        if candidate:
            proposed = self.get(candidate)
            entries = [
                item
                for item in entries
                if {item["cas1"], item["cas2"]} != {proposed["cas1"], proposed["cas2"]}
            ] + [proposed]
        records = []
        for item in entries:
            full = self.get(item["id"]) if "result" not in item else item
            result = full["result"]
            if not all(
                isinstance(cas, str) and re.fullmatch(r"\d{2,7}-\d{2}-\d", cas)
                for cas in (full["cas1"], full["cas2"])
            ):
                raise ValueError(
                    "Runtime inclusion requires two resolved CAS identities."
                )
            params = {
                key: val
                for key, val in result["parameters"].items()
                if key != "comment"
            }
            records.append(
                {
                    **params,
                    "cas1": full["cas1"],
                    "cas2": full["cas2"],
                    "component1": result["component_names"][0],
                    "component2": result["component_names"][1],
                    "source": full["source"]["citation"],
                    "source_file": "data/source/activity_fitting/user_activity_fits.sqlite",
                    "user_fit_id": full["id"],
                    "comment": full["review_notes"]
                    or "Administrator-published experimental fit",
                    "fit_status": "admin_published_user_fit",
                    "fit_vapor_treatment": {"type": result["request"]["vapor"]},
                    "fit_provenance": {
                        "source": full["source"],
                        "owner": full["owner"],
                        "reviewed_by": full["reviewed_by"],
                        "created": full["created"],
                        "model": result["method"],
                    },
                }
            )
        # Publication evaluates a candidate appended to existing fits, whereas
        # a full rebuild starts from newest-first database listings. Export a
        # single stable order so both paths produce identical runtime tables.
        return sorted(records, key=lambda record: (
            record['cas1'], record['cas2'], record['user_fit_id'],
        ))
