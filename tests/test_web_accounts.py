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


def account(client, name="PFDSim", mode="register", **credentials):
    response = client.post(
        "/api/account/" + mode,
        json={"username": name, "password": "a long laboratory password", **credentials},
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


def test_root_has_unlimited_daily_usage_and_new_jobs(clients, monkeypatch):
    first, second = clients
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = web.jobs()
    registered = account(
        first, "ROOT", setup_token=store.root_token_path.read_text().strip()
    )
    principal = "user:" + registered["user"]["id"]
    assert registered["quota"]["limit_seconds"] is None
    assert registered["quota"]["remaining_seconds"] is None
    assert registered["quota"]["grace_seconds"] == 0
    assert not store.charge(principal, 900, 1800)
    quota = first.get("/api/session").get_json()["quota"]
    assert quota["used_seconds"] == 1800
    assert quota["limit_seconds"] is None
    assert quota["remaining_seconds"] is None
    response = first.post("/api/simulate", json={"text": "PROCESS: Unlimited\n"})
    assert response.status_code == 202, response.get_json()
    assert response.get_json()["quota"] == quota
    with store.connect() as db:
        assert db.execute(
            "SELECT cpu_limit FROM jobs WHERE id=?",
            (response.get_json()["job_id"],),
        ).fetchone()[0] is None
    signed_in = account(second, "root", mode="login")
    assert signed_in["quota"] == quota
    logged_out = first.post("/api/account/logout", json={}).get_json()
    assert logged_out["quota"]["limit_seconds"] == 300
    assert logged_out["quota"]["remaining_seconds"] == 300
    assert logged_out["quota"]["grace_seconds"] == 0


def test_non_root_admin_and_unknown_principals_remain_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path)
    user = store.register("rootish", "a long laboratory password", "guest:admin")
    with store.connect() as db:
        db.execute("INSERT INTO account_roles VALUES (?, 'admin')", (user["id"],))
    assert store.user(user["id"])["is_admin"]
    for principal in ("user:" + user["id"], "root", "user:root", "guest:root"):
        assert store.quota(principal, 900)["limit_seconds"] == 900
        assert store.quota(principal, 900)["grace_seconds"] == (
            15 if principal == "user:" + user["id"] else 0
        )
        assert store.charge(principal, 900, 900)
        assert store.quota(principal, 900)["remaining_seconds"] == 0
        with pytest.raises(AccessLimit):
            store.submit("simulation", {}, principal=principal, cpu_limit=900)


def test_account_cpu_grace_does_not_allow_new_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path)
    user = store.register("PFDSim", "a long laboratory password", "guest:account")
    principal = "user:" + user["id"]
    assert not store.charge(principal, 900, 900, allow_grace=True)
    assert store.quota(principal, 900)["remaining_seconds"] == 0
    assert store.quota(principal, 900)["limit_seconds"] == 900
    with pytest.raises(AccessLimit):
        store.submit("simulation", {}, principal=principal)
    assert not store.charge(principal, 900, 14.5, allow_grace=True)
    assert store.charge(principal, 900, 0.5, allow_grace=True)
    assert store.quota(principal, 900)["used_seconds"] == 915
    # Guest and unknown principals cannot acquire account grace.
    assert store.charge("guest:account", 300, 300, allow_grace=True)
    assert store.charge("user:unknown", 900, 900, allow_grace=True)


@pytest.mark.parametrize("name,used_before", [("root", 1200), ("PFDSim", 0)])
def test_worker_finishes_with_root_exemption_or_account_grace(
    tmp_path, monkeypatch, name, used_before
):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path)
    credentials = {"setup_token": store.root_token_path.read_text().strip()} if name == "root" else {}
    user = store.register(name, "a long laboratory password", "guest:account", **credentials)
    principal = "user:" + user["id"]
    assert not store.charge(principal, 900, used_before)
    identifier = store.submit("simulation", {}, principal=principal)
    with store.connect() as db:
        # A queued job created before the exemption still has a finite limit.
        db.execute("UPDATE jobs SET cpu_limit=0.15 WHERE id=?", (identifier,))
    script = """import sys,time
import web_jobs
def busy(*args):
    start = time.process_time()
    while time.process_time() - start < .3: pass
    time.sleep(.2)
    return {'completed': True}
web_jobs.calculate_simulation = busy
web_jobs.run_calculation(web_jobs.JobStore(sys.argv[1]), sys.argv[2], {})
"""
    with (tmp_path / "grace-cpu-worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), identifier],
            stdout=log, stderr=log,
        )
    try:
        assert child.wait(timeout=10) == 0
        job = store.get(identifier)
        assert job["status"] == "completed", job
        assert job["output"] == {"completed": True}
        assert job["cpu_seconds"] >= .3
        quota = store.quota(principal, .15)
        assert quota["used_seconds"] >= used_before + .3
        assert quota["remaining_seconds"] == (None if name == "root" else 0)
        if name != "root":
            with pytest.raises(AccessLimit):
                store.submit("simulation", {}, principal=principal, cpu_limit=.15)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("name", [None, "PFDSim", "root"])
def test_job_start_reports_fresh_quota_for_budget_warning(clients, monkeypatch, name):
    first, _ = clients
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = web.jobs()
    if name:
        credentials = {"setup_token": store.root_token_path.read_text().strip()} if name == "root" else {}
        signed_in = account(first, name, **credentials)
        principal, limit = "user:" + signed_in["user"]["id"], 900
    else:
        with web.app.test_request_context(environ_base={"REMOTE_ADDR": "127.0.0.1"}):
            principal = web.guest_principal()
        limit = 300
    store.charge(principal, limit, limit - 59)
    response = first.post("/api/simulate", json={"text": "PROCESS: Low budget\n"})
    assert response.status_code == 202, response.get_json()
    quota = response.get_json()["quota"]
    assert quota["remaining_seconds"] == (None if name == "root" else 59)
    assert quota["used_seconds"] == limit - 59


@pytest.mark.parametrize("name", ["PFDSim", "root"])
def test_usage_history_is_private_recent_and_bounded(clients, monkeypatch, name):
    first, second = clients
    store = web.jobs()
    credentials = {"setup_token": store.root_token_path.read_text().strip()} if name == "root" else {}
    signed_in = account(first, name, **credentials)
    other = account(second, "OtherLab")
    owner = principal = "user:" + signed_in["user"]["id"]
    other_owner = "user:" + other["user"]["id"]
    now = time.time()
    records = [
        (f"own-{i}", owner, principal, now - i, ("running", "failed", "cancelled", "completed", "queued")[i % 5], i + 1)
        for i in range(7)
    ]
    records.extend([
        ("old", owner, principal, now - 86401, "completed", 100),
        ("other", other_owner, other_owner, now, "completed", 100),
        ("guest-claimed", owner, "guest:shared-ip", now, "completed", 100),
    ])
    with store.connect() as db:
        db.executemany(
            """INSERT INTO jobs (id, owner, principal, created, updated, status, cpu_seconds, kind, payload, output)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'simulation', 'private input', 'private output')""",
            [(identifier, row_owner, row_principal, created, created, status, cpu)
             for identifier, row_owner, row_principal, created, status, cpu in records],
        )
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: pytest.fail("Usage inspection must not start workers"))
    response = first.get("/api/usage?owner=" + other_owner)
    assert response.status_code == 200
    jobs = response.get_json()["recent_jobs"]
    assert [job["cpu_seconds"] for job in jobs] == [1, 2, 3, 4, 5]
    assert [job["status"] for job in jobs] == ["running", "failed", "cancelled", "completed", "queued"]
    assert all(set(job) == {"kind", "status", "created", "cpu_seconds"} for job in jobs)
    assert second.get("/api/usage").get_json()["recent_jobs"][0]["cpu_seconds"] == 100
    with store.connect() as db:
        assert db.execute("SELECT status FROM jobs WHERE id='own-0'").fetchone()[0] == "running"


def test_guest_usage_history_is_session_private_and_not_account_billed(clients, monkeypatch):
    first, second = clients
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    first_id = first.post("/api/simulate", json={"text": "PROCESS: First guest\n"}).get_json()["job_id"]
    second_id = second.post("/api/simulate", json={"text": "PROCESS: Second guest\n"}).get_json()["job_id"]
    store = web.jobs()
    store.update(first_id, status="completed", cpu_seconds=2)
    store.update(second_id, status="completed", cpu_seconds=3)
    assert [job["cpu_seconds"] for job in first.get("/api/usage").get_json()["recent_jobs"]] == [2]
    assert [job["cpu_seconds"] for job in second.get("/api/usage").get_json()["recent_jobs"]] == [3]
    account(first)
    assert first.get("/api/usage").get_json()["recent_jobs"] == []
    assert first.get("/api/jobs/" + first_id).status_code == 200


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


@pytest.mark.parametrize("name", ["PFDSim", "root"])
def test_guest_runs_follow_account_signin_without_moving_cpu_charges(clients, monkeypatch, name):
    first, second = clients
    monkeypatch.setattr(JobStore, 'ensure_worker', lambda *args: None)
    identifier = first.post('/api/simulate', json={'text': 'PROCESS: Guest work\n'}).get_json()['job_id']
    credentials = {"setup_token": web.jobs().root_token_path.read_text().strip()} if name == "root" else {}
    signed_in = account(first, name, **credentials)
    account(second, name, mode='login')
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
