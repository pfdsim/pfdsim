"""Bounded, cross-worker web calculations with durable status and cancellation.

SQLite coordinates a small persistent process pool across Gunicorn workers.
Canonical flowsheets have stable affinity and reuse their initialized simulator.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import resource
import signal

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .web_storage import WebStore, AccessLimit
else:
    from web_storage import WebStore, AccessLimit

TERMINAL = frozenset({"completed", "failed", "cancelled"})
PROC_ROOT = Path(Path.cwd().anchor) / "proc"


def worker_alive(pid, token):
    """A reused PID is not proof that our calculation interpreter is alive."""
    if not pid or not token:
        return False
    try:
        arguments = (PROC_ROOT / str(pid) / "cmdline").read_bytes().split(b"\0")
        return (
            str(Path(__file__).resolve()).encode() in arguments
            and b"--token" in arguments
            and arguments[arguments.index(b"--token") + 1] == token.encode()
        )
    except (OSError, IndexError):
        return False


class ComputationMeter:
    """CPU in this calculation process and its subprocesses, excluding waits.

    RUSAGE_CHILDREN covers reaped children; /proc adds still-running children.
    Only the dedicated calculation workers use this meter, never HTTP workers.
    """

    def __init__(self):
        self.ticks = os.sysconf("SC_CLK_TCK")
        self.baseline = 0.0

    @staticmethod
    def descendants():
        pending = [os.getpid()]
        descendants = {}
        while pending:
            parent = pending.pop()
            children = set()
            for file in (PROC_ROOT / str(parent) / "task").glob("*/children"):
                try:
                    children.update(int(value) for value in file.read_text().split())
                except (OSError, ValueError):
                    continue
            for child in children:
                if child in descendants:
                    continue
                try:
                    stat = (
                        (PROC_ROOT / str(child) / "stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()
                    )
                    descendants[child] = (
                        int(stat[19]),
                        sum(int(stat[index]) for index in (11, 12, 13, 14)),
                    )
                    pending.append(child)
                except (OSError, ValueError, IndexError):
                    continue
        return descendants

    def total(self):
        own = resource.getrusage(resource.RUSAGE_SELF)
        reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (
            own.ru_utime
            + own.ru_stime
            + reaped.ru_utime
            + reaped.ru_stime
            + sum(value[1] for value in self.descendants().values()) / self.ticks
        )

    def start(self):
        self.baseline = self.total()

    def cpu_seconds(self):
        return max(0.0, self.total() - self.baseline)

    def stop_children(self):
        for pid in reversed(list(self.descendants())):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def finite_json(value):
    """JSON cannot represent NaN/infinity; unavailable results become null."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return finite_json(value.tolist())
    return value


class JobStore(WebStore):
    def __init__(self, directory, *, capacity=2, deadline=1800):
        super().__init__(directory)
        self.capacity = capacity
        self.deadline = deadline
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT, status TEXT, created REAL,
                updated REAL, payload TEXT, output TEXT, progress TEXT,
                error TEXT, pid INTEGER, slot INTEGER,
                owner TEXT, principal TEXT, cpu_limit REAL, cpu_seconds REAL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS workers (
                slot INTEGER PRIMARY KEY, token TEXT, pid INTEGER, heartbeat REAL)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS affinities (fingerprint TEXT PRIMARY KEY, slot INTEGER)"
            )

    def get(self, identifier):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        # A killed worker must not occupy a slot indefinitely. A short grace
        # period allows a newly launched interpreter to register its PID.
        if job["status"] == "running" and time.time() - job["updated"] > 15:
            with self.connect() as db:
                worker = db.execute("SELECT token FROM workers WHERE slot=?", (job["slot"],)).fetchone()
            alive = worker_alive(job["pid"], worker[0] if worker else None)
            if not alive:
                self.update(
                    identifier,
                    status="failed",
                    error="Calculation worker stopped or exceeded its time limit.",
                )
                return self.get(identifier)
        elif job["status"] == "queued":
            self.ensure_worker(job["slot"])
        job["output"] = json.loads(job["output"]) if job["output"] else None
        job["progress"] = json.loads(job["progress"] or "[]")
        job.pop("payload")
        job.pop("pid")
        job.pop("principal")
        job.pop("cpu_limit")
        return job

    def update(self, identifier, **values):
        allowed = {"status", "output", "progress", "error", "pid", "cpu_seconds"}
        if not values.keys() <= allowed:
            raise ValueError("Unknown job field")
        terminal_guard = (
            ""
            if values.keys() == {"cpu_seconds"}
            else " AND status NOT IN ('completed','failed','cancelled')"
        )
        values["updated"] = time.time()
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE id=?"
                + terminal_guard,
                (*values.values(), identifier),
            )

    def submit(
        self, kind, payload, *, owner="internal", principal="internal", cpu_limit=900
    ):
        # Reap dead children before admission. The transaction then serializes
        # admission across every HTTP worker.
        if self.quota(principal, cpu_limit)["remaining_seconds"] <= 0:
            raise AccessLimit(
                "Your daily computation allowance is exhausted. It resets at 00:00 UTC."
            )
        with self.connect() as db:
            ids = db.execute(
                "SELECT id FROM jobs WHERE status IN ('queued','running')"
            ).fetchall()
        for (identifier,) in ids:
            self.get(identifier)
        fingerprint = (
            payload.get("fingerprint")
            or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        )
        identifier = secrets.token_urlsafe(24)
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if count >= self.capacity * 4:
                raise RuntimeError(
                    "The calculation queue is full. Wait for a run to finish or cancel it."
                )
            affinity = db.execute(
                "SELECT slot FROM affinities WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            if affinity:
                slot = affinity[0]
            else:
                active = dict(
                    db.execute(
                        "SELECT slot,count(*) FROM jobs WHERE status IN ('queued','running') GROUP BY slot"
                    )
                )
                slot = min(
                    range(self.capacity),
                    key=lambda value: (active.get(value, 0), value),
                )
                db.execute("INSERT INTO affinities VALUES (?,?)", (fingerprint, slot))
            db.execute(
                "DELETE FROM jobs WHERE status IN ('completed','failed','cancelled') AND updated < ?",
                (now - 86400,),
            )
            db.execute(
                "INSERT INTO jobs (id,kind,status,created,updated,payload,progress,slot,owner,principal,cpu_limit) VALUES (?,?,'queued',?,?,?,'[]',?,?,?,?)",
                (
                    identifier,
                    kind,
                    now,
                    now,
                    json.dumps(payload, allow_nan=False),
                    slot,
                    owner,
                    principal,
                    cpu_limit,
                ),
            )
        try:
            self.ensure_worker(slot)
        except Exception:
            self.update(
                identifier, status="failed", error="Could not start calculation worker."
            )
            raise
        return identifier

    def ensure_worker(self, slot):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT token,pid,heartbeat FROM workers WHERE slot=?", (slot,)
            ).fetchone()
            if row:
                if row[1] is None and time.time() - row[2] < 10:
                    return
                if row[1] is not None:
                    if worker_alive(row[1], row[0]):
                        return
            token = secrets.token_urlsafe(16)
            db.execute(
                "INSERT OR REPLACE INTO workers VALUES (?,?,NULL,?)",
                (slot, token, time.time()),
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--directory",
            str(self.directory),
            "--slot",
            str(slot),
            "--token",
            token,
            "--deadline",
            str(self.deadline),
        ]
        try:
            with (self.directory / "workers.log").open("ab") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=log, start_new_session=True
                )
            with self.connect() as db:
                db.execute(
                    "UPDATE workers SET pid=? WHERE slot=? AND token=?",
                    (child.pid, slot, token),
                )
            threading.Thread(target=child.wait, daemon=True).start()
        except Exception:
            with self.connect() as db:
                db.execute(
                    "DELETE FROM workers WHERE slot=? AND token=?", (slot, token)
                )
            raise

    def cancel(self, identifier):
        job = self.get(identifier)
        if job is not None and job["status"] not in TERMINAL:
            self.update(identifier, status="cancelled", error="Cancelled by you.")
        return self.get(identifier)


def initialized_simulator(payload, progress, simulators):
    # The worker runs as a script from the package directory.
    from simulator import Simulator

    fingerprint = payload["fingerprint"]
    reused = fingerprint in simulators
    if reused:
        sim = simulators.pop(fingerprint)
    else:
        sim = Simulator.from_string(payload["text"])
    simulators[fingerprint] = sim
    while len(simulators) > 2:
        simulators.popitem(last=False)
    progress("Reusing initialized flowsheet" if reused else "Initializing flowsheet")
    sim.initialize()
    return sim, reused


def calculate_simulation(payload, progress, simulators):
    sim, reused = initialized_simulator(payload, progress, simulators)
    result = sim.run(
        max_iterations=payload["max_iterations"],
        tolerance=payload["tolerance"],
        progress_callback=progress,
    )
    return {
        "success": True,
        "converged": result.converged,
        "iterations": result.iterations,
        "mass_balance_error": result.mass_balance_error,
        "energy_balance_error": result.energy_balance_error,
        "thermo_method": sim.thermo_method,
        "pfr_content": sim._generate_pfr(),
        "results": sim.get_results_dict(),
        "errors": result.errors,
        "warnings": result.warnings,
        "reused_flowsheet": reused,
    }


def calculate_chart(payload, progress, simulators):
    sim, reused = initialized_simulator(payload, progress, simulators)
    thermo = sim.thermo_packages[payload["scope"]]
    components = payload["components"]
    identities = [
        thermo.props[c].CAS or thermo.props[c].name.lower() for c in components
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "The identifiers resolve to the same chemical. Choose distinct components."
        )
    chart_type = payload["chart_type"]
    progress(f"Calculating {chart_type} with {payload['method']}")
    if chart_type == "TERNARY_LLE":
        chart = thermo.generate_ternary_lle_data(
            components, payload["temperature"] + 273.15, payload["n_points"], progress
        )
    elif chart_type == "VLLE":
        chart = thermo.generate_binary_vlle_data(
            *components, P=payload['pressure'], n_points=payload['n_points'],
            minimum_temperature=payload.get('minimum_temperature',25)+273.15,
            progress=progress,
        )
    elif chart_type == "TERNARY_VLLE":
        chart = thermo.generate_vlle_data(
            components,
            payload["temperature"] + 273.15,
            payload["pressure"],
            payload["n_points"],
            progress,
        )
    elif chart_type == "PXY":
        chart = thermo.generate_Pxy_data(
            *components, payload["temperature"] + 273.15, payload["n_points"]
        )
    elif chart_type == "TXY":
        chart = thermo.generate_Txy_data(
            *components, P=payload["pressure"], n_points=payload["n_points"]
        )
    else:
        chart = thermo.generate_xy_data(
            *components, P=payload["pressure"], n_points=payload["n_points"]
        )
    return {
        "success": True,
        **chart,
        "chart_type": chart_type,
        **{f"comp{i + 1}": c for i, c in enumerate(components)},
        **{f"comp{i + 1}_name": thermo.props[c].name for i, c in enumerate(components)},
        "pressure_bar": payload["pressure"],
        "temperature_C": payload["temperature"],
        "method": payload["method"],
        "reused_flowsheet": reused,
        "scope": payload["scope"],
    }


def calculate_groups(payload, progress, simulators=None):
    from chemical_properties import get_database
    from unifac import get_unifac_groups, parse_smiles_to_unifac, RDKIT_AVAILABLE

    identifier, variant = payload["identifier"], payload["variant"]
    progress(f"Resolving UNIFAC groups for {identifier}")
    if payload.get("smiles"):
        groups = parse_smiles_to_unifac(payload["smiles"], variant)
        source = "native_smiles"
    else:
        try:
            groups = get_unifac_groups(identifier, variant=variant)
            source = "curated"
        except ValueError:
            info = get_database().resolve_smiles_info(
                identifier, fetch_online=payload["online_lookup"]
            )
            if not info or not info.smiles:
                raise ValueError(
                    f"Could not resolve UNIFAC groups for {identifier!r}"
                ) from None
            groups = get_unifac_groups(identifier, smiles=info.smiles, variant=variant)
            source = "resolved_smiles"
    if groups is None:
        raise ValueError(f"Could not resolve UNIFAC groups for {identifier!r}")
    return {
        "success": True,
        "identifier": identifier,
        "groups": groups,
        "source": source,
        "rdkit_available": RDKIT_AVAILABLE,
    }


def run_calculation(store, identifier, simulators):
    with store.connect() as db:
        row = db.execute(
            "SELECT kind,payload,status,principal,cpu_limit FROM jobs WHERE id=?",
            (identifier,),
        ).fetchone()
    if row is None or row[2] in TERMINAL:
        return
    kind, raw_payload, _, principal, cpu_limit = row
    if store.quota(principal, cpu_limit)["remaining_seconds"] <= 0:
        store.update(
            identifier,
            status="failed",
            error="Your daily computation allowance is exhausted. It resets at 00:00 UTC.",
        )
        return
    store.update(identifier, status="running", pid=os.getpid())
    done = threading.Event()
    started = time.monotonic()
    meter = ComputationMeter()
    meter.start()
    accounting_lock = threading.Lock()
    charged = 0.0

    def account():
        nonlocal charged
        with accounting_lock:
            current = meter.cpu_seconds()
            exhausted = store.charge(principal, cpu_limit, max(0.0, current - charged))
            charged = current
            store.update(identifier, cpu_seconds=current)
            return exhausted

    def monitor():
        while not done.wait(0.1):
            if account():
                store.update(
                    identifier,
                    status="failed",
                    error="Daily CPU allowance exhausted during this calculation. It resets at 00:00 UTC.",
                )
                meter.stop_children()
                os._exit(0)
            with store.connect() as db:
                state = db.execute(
                    "SELECT status FROM jobs WHERE id=?", (identifier,)
                ).fetchone()
            if state is None or state[0] == "cancelled":
                meter.stop_children()
                os._exit(0)
            if time.monotonic() - started > store.deadline:
                store.update(
                    identifier,
                    status="failed",
                    error="Calculation exceeded the 30-minute time limit.",
                )
                meter.stop_children()
                os._exit(0)
            store.update(identifier)

    threading.Thread(target=monitor, daemon=True).start()
    messages = []

    def progress(message):
        messages.append(str(message))
        store.update(identifier, progress=json.dumps(messages[-100:]))

    try:
        calculate = {
            "simulation": calculate_simulation,
            "chart": calculate_chart,
            "groups": calculate_groups,
        }[kind]
        output = calculate(json.loads(raw_payload), progress, simulators)
        if account():
            store.update(
                identifier,
                status="failed",
                error="Daily CPU allowance exhausted during this calculation. It resets at 00:00 UTC.",
            )
        else:
            store.update(
                identifier,
                status="completed",
                output=json.dumps(finite_json(output), allow_nan=False),
            )
    except Exception as error:
        import traceback

        traceback.print_exc()  # Private worker log, never an HTTP response.
        store.update(identifier, status="failed", error=str(error))
    finally:
        done.set()
        account()


def run_worker(store, slot, token):
    simulators = OrderedDict()
    idle_since = time.monotonic()
    while time.monotonic() - idle_since < 900:
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = db.execute(
                "SELECT token FROM workers WHERE slot=?", (slot,)
            ).fetchone()
            if worker is None or worker[0] != token:
                return
            db.execute(
                "UPDATE workers SET heartbeat=?,pid=? WHERE slot=? AND token=?",
                (time.time(), os.getpid(), slot, token),
            )
            row = db.execute(
                "SELECT id FROM jobs WHERE slot=? AND status='queued' ORDER BY created LIMIT 1",
                (slot,),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE jobs SET status='running',pid=?,updated=? WHERE id=? AND status='queued'",
                    (os.getpid(), time.time(), row[0]),
                )
        if row:
            idle_since = time.monotonic()
            run_calculation(store, row[0], simulators)
        else:
            time.sleep(0.4)
    with store.connect() as db:
        db.execute("DELETE FROM workers WHERE slot=? AND token=?", (slot, token))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--deadline", type=float, default=1800)
    args = parser.parse_args()
    run_worker(JobStore(args.directory, deadline=args.deadline), args.slot, args.token)
