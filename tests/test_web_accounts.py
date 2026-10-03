"""Accounts, private autosaves, and actual computation CPU budgets."""

import subprocess
import sys
import time

import pytest

pytest.importorskip("flask")
import app as web
from pfd_parser import ProcessFlowDiagram
from web_jobs import JobStore, ComputationMeter
from web_storage import WebStore, AccessLimit


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setitem(web.app.config, "TESTING", True)
    monkeypatch.setitem(web.app.config, "JOB_DIRECTORY", tmp_path)
    first, second = web.app.test_client(), web.app.test_client()
    for client in (first, second):
        session = client.get("/api/session").get_json()
        client.environ_base["HTTP_X_CSRF_TOKEN"] = session["csrf_token"]
    return first, second


def account(client, name="PFDSim", mode="register"):
    response = client.post(
        "/api/account/" + mode,
        json={"username": name, "password": "a long laboratory password"},
    )
    assert response.status_code == 200, response.get_json()
    client.environ_base["HTTP_X_CSRF_TOKEN"] = response.get_json()["csrf_token"]
    return response.get_json()


def test_accounts_hash_passwords_and_casefold_usernames(clients):
    first, second = clients
    signed_in = account(first)
    assert signed_in["quota"]["limit_seconds"] == 900
    with web.jobs().connect() as db:
        stored = db.execute("SELECT password_hash FROM users").fetchone()[0]
    assert stored.startswith("scrypt:")
    assert "a long laboratory password" not in stored
    assert (
        second.post(
            "/api/account/register",
            json={"username": "PFDSim", "password": "another long password"},
        ).status_code
        == 409
    )
    assert (
        second.post(
            "/api/account/login", json={"username": "PFDSim", "password": "incorrect"}
        ).status_code
        == 400
    )
    assert (
        account(second, "PFDSim", mode="login")["user"]["id"] == signed_in["user"]["id"]
    )
    response = first.post("/api/account/logout", json={})
    assert response.get_json()["user"] is None
    assert response.get_json()["quota"]["limit_seconds"] == 300


def test_account_mutations_require_csrf(clients):
    first, _ = clients
    first.environ_base.pop("HTTP_X_CSRF_TOKEN")
    assert (
        first.post(
            "/api/account/register",
            json={"username": "PFDSim", "password": "a long laboratory password"},
        ).status_code
        == 403
    )
    assert (
        first.post("/api/simulate", json={"text": "PROCESS: test\n"}).status_code == 403
    )


def test_autosaved_text_drafts_are_private_and_versioned(clients):
    first, second = clients
    account(first)
    account(second, "AnotherLab")
    document = {
        "pfd": ProcessFlowDiagram().to_dict(),
        "text": "unfinished invalid draft",
        "pending": True,
        "filename": "draft.pfd",
    }
    response = first.post(
        "/api/flowsheets/laboratory-001", json={"version": 0, "document": document}
    )
    assert response.status_code == 200
    assert response.get_json()["version"] == 1
    saved = first.get("/api/flowsheets").get_json()["flowsheets"][0]
    assert saved["document"]["text"] == document["text"]
    assert saved["document"]["pending"] is True
    assert second.get("/api/flowsheets").get_json()["flowsheets"] == []
    assert (
        second.post(
            "/api/flowsheets/laboratory-001", json={"version": 0, "document": document}
        ).status_code
        == 409
    )
    assert (
        first.post(
            "/api/flowsheets/laboratory-001", json={"version": 0, "document": document}
        ).status_code
        == 409
    )
    assert (
        first.get("/api/flowsheets").get_json()["flowsheets"][0]["document"]["text"]
        == document["text"]
    )
    document["text"] = "PROCESS: finished\n"
    assert (
        first.post(
            "/api/flowsheets/laboratory-001", json={"version": 1, "document": document}
        ).get_json()["version"]
        == 2
    )


def test_jobs_cannot_be_read_or_cancelled_by_another_visitor(clients, monkeypatch):
    first, second = clients
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    response = first.post("/api/simulate", json={"text": "PROCESS: private\n"})
    assert response.status_code == 202
    identifier = response.get_json()["job_id"]
    assert first.get("/api/jobs/" + identifier).status_code == 200
    assert second.get("/api/jobs/" + identifier).status_code == 404
    assert (
        second.post("/api/jobs/" + identifier + "/cancel", json={}).status_code == 404
    )
    assert first.get("/api/jobs/" + identifier).get_json()["job"]["status"] == "queued"


def test_guest_budget_survives_clearing_cookies_and_ignores_untrusted_forwarded_ip(
    clients, monkeypatch
):
    first, second = clients
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    with web.app.test_request_context(environ_base={"REMOTE_ADDR": "127.0.0.1"}):
        principal = web.guest_principal()
    web.jobs().charge(principal, 300, 300)
    assert first.get("/api/session").get_json()["quota"]["remaining_seconds"] == 0
    assert (
        second.get("/api/session", headers={"X-Forwarded-For": "192.0.2.5"}).get_json()[
            "quota"
        ]["remaining_seconds"]
        == 0
    )
    assert (
        first.post(
            "/api/simulate", json={"text": "PROCESS: no allowance\n"}
        ).status_code
        == 429
    )
    registered = account(first)
    assert registered["quota"]["remaining_seconds"] == 900


def test_daily_accounting_is_atomic_across_store_instances_and_resets(
    tmp_path, monkeypatch
):
    first, second = WebStore(tmp_path), WebStore(tmp_path)
    monkeypatch.setattr(WebStore, "day", staticmethod(lambda: "2026-10-02"))
    assert not first.charge("person", 300, 140)
    assert second.charge("person", 300, 160)
    assert first.quota("person", 300)["used_seconds"] == 300
    monkeypatch.setattr(WebStore, "day", staticmethod(lambda: "2026-10-03"))
    assert second.quota("person", 300)["remaining_seconds"] == 300


def test_cpu_meter_excludes_waiting_and_counts_running_cpu():
    meter = ComputationMeter()
    meter.start()
    time.sleep(0.2)
    waited = meter.cpu_seconds()
    assert waited < 0.08
    start = time.process_time()
    while time.process_time() - start < 0.12:
        pass
    assert meter.cpu_seconds() - waited >= 0.1


def test_worker_stops_when_real_cpu_budget_is_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path, capacity=1)
    identifier = store.submit("simulation", {}, principal="tiny-budget", cpu_limit=0.15)
    with store.connect() as db:
        db.execute("INSERT INTO workers VALUES (0,?,NULL,?)", ("cpu-test", time.time()))
    script = """import sys,time
import web_jobs
def busy(*args):
    while True: pass
web_jobs.calculate_simulation=busy
web_jobs.run_worker(web_jobs.JobStore(sys.argv[1],capacity=1),0,'cpu-test')
"""
    with (tmp_path / "cpu-worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)], stdout=log, stderr=log
        )
    try:
        assert child.wait(timeout=10) == 0
        job = store.get(identifier)
        assert job["status"] == "failed"
        assert "CPU allowance" in job["error"]
        assert store.quota("tiny-budget", 0.15)["used_seconds"] >= 0.15
        with pytest.raises(AccessLimit):
            store.submit("simulation", {}, principal="tiny-budget", cpu_limit=0.15)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_new_flowsheets_use_idle_worker_but_same_flowsheet_keeps_affinity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path, capacity=2)
    first = store.submit("simulation", {"fingerprint": "aaaa"})
    second = store.submit("simulation", {"fingerprint": "bbbb"})
    repeat = store.submit("simulation", {"fingerprint": "aaaa"})
    assert store.get(first)["slot"] != store.get(second)["slot"]
    assert store.get(repeat)["slot"] == store.get(first)["slot"]


def test_guest_runs_follow_account_signin_without_moving_cpu_charges(clients, monkeypatch):
    first, second = clients
    monkeypatch.setattr(JobStore, 'ensure_worker', lambda *args: None)
    identifier = first.post('/api/simulate', json={'text': 'PROCESS: Guest work\n'}).get_json()['job_id']
    signed_in = account(first)
    account(second, mode='login')
    assert second.get('/api/jobs/'+identifier).status_code == 200
    with web.jobs().connect() as db:
        row = db.execute('SELECT owner,principal,cpu_limit FROM jobs WHERE id=?', (identifier,)).fetchone()
    assert row[0] == 'user:'+signed_in['user']['id']
    assert row[1].startswith('guest:')
    assert row[2] == 300


def test_running_subprocess_cpu_is_counted(tmp_path):
    meter = ComputationMeter()
    meter.start()
    script = 'import time\nstart=time.process_time()\nwhile time.process_time()-start<.3: pass\ntime.sleep(.5)\n'
    with (tmp_path/'meter-child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', script], stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+3
        while meter.cpu_seconds()<.15 and time.monotonic()<deadline:
            time.sleep(.025)
        assert meter.cpu_seconds() >= .15
        child.wait(timeout=5)
        assert meter.cpu_seconds() >= .3
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
