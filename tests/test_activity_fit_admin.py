"""Protected root signup, durable review, builder publication and recovery."""

from copy import deepcopy
from contextlib import contextmanager
import json

import pytest

import app as web
from activity_fit_store import ActivityFitStore
from web_jobs import JobStore
from scripts import build_cas_interaction_parameters as builder
from thermodynamics_models.interaction_fitting import (
    fit_interactions,
    validate_runtime_inclusion,
)
from .test_interaction_fitting import synthetic


@pytest.fixture(scope="module")
def fitted():
    data, _, _ = synthetic(kinds=("GAMMA_INF", "HE"))
    return fit_interactions(data)


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setitem(web.app.config, "TESTING", True)
    monkeypatch.setitem(web.app.config, "JOB_DIRECTORY", tmp_path)
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    first, second = web.app.test_client(), web.app.test_client()
    for client in (first, second):
        token = client.get("/api/session").get_json()["csrf_token"]
        client.environ_base["HTTP_X_CSRF_TOKEN"] = token
    return first, second


def root_signup(client):
    token = web.jobs().root_token_path.read_text().strip()
    response = client.post(
        "/api/account/register",
        json={
            "username": "root",
            "password": "long administrator password",
            "setup_token": token,
        },
    )
    assert response.status_code == 200, response.get_json()
    client.environ_base["HTTP_X_CSRF_TOKEN"] = response.get_json()["csrf_token"]
    return response.get_json()["user"]


def test_root_requires_one_time_token_and_role_is_authenticated(clients):
    root, guest = clients
    denied = root.post(
        "/api/account/register",
        json={"username": "root", "password": "long administrator password"},
    )
    assert denied.status_code == 400
    assert "one-time setup token" in denied.get_json()["error"]
    assert guest.get("/api/fitting/admin/submissions").status_code == 403
    user = root_signup(root)
    assert user["is_admin"] is True
    assert root.get("/api/fitting/admin/submissions").status_code == 200
    assert (
        guest.post(
            "/api/account/register",
            json={
                "username": "ROOT",
                "password": "another long password",
                "setup_token": web.jobs().root_token_path.read_text().strip(),
            },
        ).status_code
        == 409
    )
    assert (
        guest.post(
            "/api/account/register",
            json={
                "username": "ordinary",
                "password": "long ordinary password",
                "is_admin": True,
            },
        ).status_code
        == 200
    )
    assert guest.get("/api/session").get_json()["user"]["is_admin"] is False


def test_review_permissions_history_versioning_and_direct_publish(clients, fitted):
    root, guest = clients
    submitted = guest.post(
        "/api/fitting/submit",
        json={
            "result": fitted,
            "source": {
                "citation": "Primary test source",
                "doi": "test-doi",
                "notes": "Complete provenance",
            },
        },
    )
    identifier = submitted.get_json()["submission"]["id"]
    root_signup(root)
    detail = root.get(f"/api/fitting/admin/submissions/{identifier}").get_json()[
        "submission"
    ]
    assert detail["result"]["request"] == fitted["request"]
    assert detail["events"][0]["action"] == "submitted"
    assert guest.get(f"/api/fitting/admin/submissions/{identifier}").status_code == 403
    assert (
        guest.post(
            "/api/fitting/admin/review",
            json={
                "id": identifier,
                "action": "approve",
                "notes": "Bypass attempt",
                "version": 1,
            },
        ).status_code
        == 403
    )
    assert (
        root.post(
            "/api/fitting/admin/review",
            json={
                "id": identifier,
                "action": "approve",
                "notes": "Reviewed measurements",
                "version": 1,
            },
        ).status_code
        == 200
    )
    assert (
        root.post(
            "/api/fitting/admin/review",
            json={
                "id": identifier,
                "action": "reject",
                "notes": "Stale decision",
                "version": 1,
            },
        ).status_code
        == 400
    )
    queued = root.post("/api/fitting/admin/publish", json={"id": identifier})
    assert queued.status_code == 202
    assert web.jobs().get(queued.get_json()["job_id"])["kind"] == "fit_publish"
    direct = root.post(
        "/api/fitting/admin/publish",
        json={"result": fitted, "source": {"citation": "Administrator direct fit"}},
    )
    assert direct.status_code == 202
    assert any(item["status"] == "approved" for item in web.activity_fit_store().list())


def test_builder_consumes_published_fits_with_full_provenance(tmp_path, fitted):
    store = ActivityFitStore(tmp_path / "user_activity_fits.sqlite")
    identifier = store.submit(
        "user:root",
        "job",
        {"citation": "Primary paper", "url": "https://example.test/source"},
        fitted,
    )["id"]
    store.review(
        identifier, "user:root", "approve", "Reviewed source tables", expected_version=1
    )
    base = {
        "metadata": {},
        "interactions": [
            {"cas1": "64-17-5", "cas2": "7732-18-5", "alpha12": 0.3, "old": True},
            {"cas1": "67-56-1", "cas2": "7732-18-5", "alpha12": 0.3},
        ],
    }
    untouched = builder.apply_user_activity_overlay(
        deepcopy(base), "NRTL", user_fits_path=store.path
    )
    assert untouched["interactions"] == base["interactions"]
    store.publication_state(identifier, "user:root", "publishing", {})
    store.publication_state(
        identifier, "user:root", "published", {"runtime_sha256": "test"}
    )
    overlaid = builder.apply_user_activity_overlay(
        deepcopy(base), "NRTL", user_fits_path=store.path
    )
    assert len(overlaid["interactions"]) == 2
    record = next(item for item in overlaid["interactions"] if item.get("user_fit_id"))
    assert record["tau12_d"] == fitted["parameters"]["tau12_d"]
    assert record["fit_provenance"]["source"]["citation"] == "Primary paper"
    assert record["fit_provenance"]["reviewed_by"] == "user:root"
    assert store.get(identifier)["result"]["points"] == fitted["points"]


def test_atomic_builder_publication_backup_withdrawal_and_supersession(
    tmp_path, fitted, monkeypatch
):
    store = ActivityFitStore(tmp_path / "source/user_activity_fits.sqlite")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "nrtl_binary_interactions_cas.json"
    base = {
        "metadata": {},
        "interactions": [
            {
                "cas1": "64-17-5",
                "cas2": "7732-18-5",
                "alpha12": 0.3,
                "a12_cal_per_mol": 1,
                "a21_cal_per_mol": 2,
            }
        ],
    }
    target.write_text(json.dumps(base))
    monkeypatch.setattr(builder, "DATA", runtime)
    monkeypatch.setattr(builder, "resolve_component_ids", lambda: ([], []))
    monkeypatch.setattr(
        builder,
        "build_interaction_payload",
        lambda source, resolved, unresolved, **settings: (
            builder.apply_user_activity_overlay(
                deepcopy(base),
                "NRTL",
                user_fits_path=settings["user_fits_path"],
                candidate=settings["user_fit_candidate"],
                exclude=settings["exclude_user_fit"],
            )
        ),
    )
    identifier = store.submit(
        "user:root", "first", {"citation": "First source"}, fitted
    )["id"]
    store.review(identifier, "user:root", "approve", "Approved", expected_version=1)
    published = builder.publish_user_fit(
        identifier, user_fits_path=store.path, actor="user:root"
    )
    assert published["status"] == "published"
    assert json.loads((store.path.parent / published["backup"]).read_text()) == base
    assert (
        json.loads(target.read_text())["interactions"][0]["user_fit_id"] == identifier
    )
    second = store.submit("user:root", "second", {"citation": "Second source"}, fitted)[
        "id"
    ]
    store.review(
        second, "user:root", "approve", "Approved replacement", expected_version=1
    )
    builder.publish_user_fit(second, user_fits_path=store.path, actor="user:root")
    assert store.get(identifier)["status"] == "superseded"
    assert store.get(second)["status"] == "published"
    builder.publish_user_fit(
        second,
        user_fits_path=store.path,
        actor="user:root",
        action="withdraw",
        notes="Withdraw test artifact",
    )
    assert json.loads(target.read_text())["interactions"] == base["interactions"]
    assert store.get(second)["status"] == "revoked"


def test_failed_builder_preserves_runtime_and_previous_publication(
    tmp_path, fitted, monkeypatch
):
    store = ActivityFitStore(tmp_path / "user_activity_fits.sqlite")
    identifier = store.submit("user:root", "job", {"citation": "Source"}, fitted)["id"]
    store.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
    store.publication_state(identifier, "user:root", "publishing", {})
    store.publication_state(identifier, "user:root", "published", {})
    monkeypatch.setattr(builder, "DATA", tmp_path)
    target = tmp_path / "nrtl_binary_interactions_cas.json"
    target.write_text('{"original":true}')
    monkeypatch.setattr(
        builder,
        "resolve_component_ids",
        lambda: (_ for _ in ()).throw(ValueError("Failed identity lookup")),
    )
    with pytest.raises(ValueError, match="Failed identity lookup"):
        builder.publish_user_fit(
            identifier, user_fits_path=store.path, actor="user:root"
        )
    assert target.read_text() == '{"original":true}'
    assert store.get(identifier)["status"] == "published"


def test_withdrawal_rechecks_status_after_waiting_for_publication_lock(
    tmp_path, fitted, monkeypatch
):
    store = ActivityFitStore(tmp_path / "user_activity_fits.sqlite")
    identifier = store.submit("user:root", "job", {"citation": "Source"}, fitted)["id"]
    store.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
    store.publication_state(identifier, "user:root", "publishing", {})
    store.publication_state(identifier, "user:root", "published", {})

    @contextmanager
    def changed_while_waiting(path):
        store.publication_state(identifier, "user:other-admin", "withdrawn", {})
        yield

    monkeypatch.setattr(builder, "DATA", tmp_path)
    monkeypatch.setattr(builder, "activity_publication_lock", changed_while_waiting)
    monkeypatch.setattr(builder, "resolve_component_ids", lambda: pytest.fail("A stale withdrawal must not rebuild runtime data"))
    with pytest.raises(ValueError, match="Only published fits"):
        builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root", action="withdraw")


def test_different_provenance_stores_lock_the_shared_runtime_destination(
    tmp_path, fitted, monkeypatch
):
    runtime = tmp_path / "runtime"
    locked = []

    @contextmanager
    def record_lock(directory):
        locked.append(directory)
        yield

    def stop_before_writes():
        raise ValueError("Stop before runtime writes")

    monkeypatch.setattr(builder, "DATA", runtime)
    monkeypatch.setattr(builder, "activity_publication_lock", record_lock)
    monkeypatch.setattr(builder, "resolve_component_ids", stop_before_writes)
    for name in ("first-source", "second-source"):
        store = ActivityFitStore(tmp_path / name / "fits.sqlite")
        identifier = store.submit("user:root", "job", {"citation": "Source"}, fitted)["id"]
        store.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
        with pytest.raises(ValueError, match="Stop before runtime writes"):
            builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root")
    assert locked == [runtime, runtime]


def test_publication_rechecks_coefficients_and_requires_portable_rq(fitted):
    assert validate_runtime_inclusion(fitted)["component_cas"] == [
        "64-17-5",
        "7732-18-5",
    ]
    broken = deepcopy(fitted)
    broken["parameters"]["tau12_d"] += 1
    with pytest.raises(ValueError, match="disagree"):
        validate_runtime_inclusion(broken)
    settings, _, _ = synthetic(
        {"model": "UNIQUAC", "rq": [{"r": 2.2, "q": 1.8}, {"r": 0.92, "q": 1.4}]},
        kinds=("GAMMA_INF",),
    )
    custom = fit_interactions(settings)
    with pytest.raises(ValueError, match="custom UNIQUAC R/Q"):
        validate_runtime_inclusion(custom)


@pytest.mark.parametrize("field,value", [
    ("Tmin_K", 123.0),
    ("Tmax_K", 456.0),
    ("extrapolation", "constant_inverse"),
    ("disabled", True),
])
def test_publication_rechecks_entire_runtime_parameter_contract(fitted, field, value):
    edited = deepcopy(fitted)
    edited["parameters"][field] = value
    with pytest.raises(ValueError, match="disagree"):
        validate_runtime_inclusion(edited)


def test_atomic_runtime_writes_preserve_read_permissions_and_cleanup(tmp_path, monkeypatch):
    target = tmp_path / "runtime.json"
    builder.atomic_write_bytes(target, b"original")
    assert target.stat().st_mode & 0o777 == 0o644
    target.chmod(0o640)
    builder.atomic_write_bytes(target, b"replacement")
    assert target.stat().st_mode & 0o777 == 0o640

    def fail_replace(*args):
        raise OSError("Replacement failed")

    monkeypatch.setattr(builder.os, "replace", fail_replace)
    with pytest.raises(OSError, match="Replacement failed"):
        builder.atomic_write_bytes(target, b"failed write")
    assert target.read_bytes() == b"replacement"
    assert not list(tmp_path.glob("*.pending"))


def test_table_generation_refreshes_new_packages(tmp_path, monkeypatch):
    import interaction_parameters as parameters
    from thermodynamics import create_thermodynamics

    for name in (
        "nrtl_binary_interactions_cas.json",
        "uniquac_binary_interactions_cas.json",
        "uniquac_rq_cas.json",
    ):
        (tmp_path / name).write_text((parameters.DATA_DIR / name).read_text())
    monkeypatch.setattr(parameters, "DATA_DIR", tmp_path)
    original = create_thermodynamics(["ethanol", "water"], "NRTL")
    # Exercise the scalar path as well: cached compiled vectors alone must not
    # be the reason an existing package survives publication consistently.
    monkeypatch.setattr(original, "_compiled_activity_backend", lambda T=None: None)
    before = original.activity_coefficients(350, {"ethanol": 0.4, "water": 0.6})
    path = tmp_path / "nrtl_binary_interactions_cas.json"
    payload = json.loads(path.read_text())
    payload["interactions"] = [
        item
        for item in payload["interactions"]
        if {item["cas1"], item["cas2"]} != {"64-17-5", "7732-18-5"}
    ]
    payload["interactions"].append(
        {
            "cas1": "64-17-5",
            "cas2": "7732-18-5",
            "alpha12": 0.3,
            "tau12_c": 0,
            "tau21_c": 0,
        }
    )
    path.write_text(json.dumps(payload))
    after = create_thermodynamics(["ethanol", "water"], "NRTL").activity_coefficients(
        350, {"ethanol": 0.4, "water": 0.6}
    )
    assert before != after
    assert after == pytest.approx({"ethanol": 1, "water": 1})
    original._activity_cache.clear()
    original._nrtl_matrix_cache.clear()
    assert original.activity_coefficients(
        350, {"ethanol": 0.4, "water": 0.6}
    ) == pytest.approx(before)


def test_interrupted_publication_can_be_explicitly_recovered(
    tmp_path, fitted, monkeypatch
):
    store = ActivityFitStore(tmp_path / "user_activity_fits.sqlite")
    identifier = store.submit(
        "user:root", "job", {"citation": "Recovery source"}, fitted
    )["id"]
    store.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
    store.publication_state(
        identifier, "user:root", "publishing", {"notes": "Interrupted run"}
    )
    monkeypatch.setattr(builder, "DATA", tmp_path)
    monkeypatch.setattr(builder, "resolve_component_ids", lambda: ([], []))
    monkeypatch.setattr(
        builder,
        "build_interaction_payload",
        lambda *args, **kwargs: {
            "metadata": {},
            "interactions": store.runtime_records("NRTL", candidate=identifier),
        },
    )
    result = builder.publish_user_fit(
        identifier,
        user_fits_path=store.path,
        actor="user:root",
        notes="Explicit recovery",
    )
    assert result["status"] == "published"
    assert (
        len(
            [
                event
                for event in store.get(identifier)["events"]
                if event["action"] == "publishing"
            ]
        )
        == 2
    )


def test_rebuild_preserves_previously_published_fit_during_interrupted_republication(
    tmp_path, fitted
):
    store = ActivityFitStore(tmp_path / "user_activity_fits.sqlite")
    identifier = store.submit("user:root", "job", {"citation": "Source"}, fitted)["id"]
    store.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
    store.publication_state(identifier, "user:root", "publishing", {})
    store.publication_state(identifier, "user:root", "published", {})
    store.publication_state(identifier, "user:root", "publishing", {})
    assert store.runtime_records("NRTL")[0]["user_fit_id"] == identifier


def test_approve_without_notes_is_saved_and_discoverable_by_name(clients, fitted):
    root, guest = clients
    submitted = guest.post(
        "/api/fitting/submit",
        json={"result": fitted, "source": {"citation": "Approval regression source"}},
    )
    identifier = submitted.get_json()["submission"]["id"]
    root_signup(root)
    response = root.post(
        "/api/fitting/admin/review",
        json={"id": identifier, "action": "approve", "version": 1},
    )
    assert response.status_code == 200, response.get_json()
    detail = response.get_json()["submission"]
    assert detail["status"] == "approved"
    assert detail["review_notes"] == "Approved by administrator."
    assert detail["events"][-1]["action"] == "approve"
    queue = root.get("/api/fitting/admin/submissions").get_json()["submissions"]
    summary = next(item for item in queue if item["id"] == identifier)
    assert summary["status"] == "approved"
    assert summary["component_names"] == fitted["component_names"]
    assert set(summary["objectives"]) == set(fitted["objectives"])
    assert detail["result"] == {
        **fitted,
        "submission_origin": "external_report_pending_review",
    }


def test_rebuilding_from_another_store_cannot_remove_published_fits(tmp_path, fitted, monkeypatch):
    first = ActivityFitStore(tmp_path / "first" / "fits.sqlite")
    identifier = first.submit("user:root", "first", {"citation": "First source"}, fitted)["id"]
    first.review(identifier, "user:root", "approve", "Reviewed", expected_version=1)
    first.publication_state(identifier, "user:root", "publishing", {})
    first.publication_state(identifier, "user:root", "published", {})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "nrtl_binary_interactions_cas.json"
    target.write_text(json.dumps({"interactions": first.runtime_records("NRTL")}))
    before = target.read_bytes()
    second = ActivityFitStore(tmp_path / "second" / "fits.sqlite")
    candidate = second.submit("user:root", "second", {"citation": "Second source"}, fitted)["id"]
    second.review(candidate, "user:root", "approve", "Reviewed", expected_version=1)
    monkeypatch.setattr(builder, "DATA", runtime)
    monkeypatch.setattr(builder, "resolve_component_ids", lambda: pytest.fail("The wrong store must be rejected before rebuilding"))
    with pytest.raises(ValueError, match="provenance store"):
        builder.publish_user_fit(candidate, user_fits_path=second.path, actor="user:root")
    with pytest.raises(ValueError, match="provenance store"):
        builder.write_interaction_file(target.name, "nrtl_binary_interactions.json", {}, [], user_fits_path=second.path)
    monkeypatch.setattr("sys.argv", ["builder", "--user-fits-path", str(second.path)])
    with pytest.raises(ValueError, match="provenance store"):
        builder.main()
    assert target.read_bytes() == before
    assert first.get(identifier)["status"] == "published"
    assert second.get(candidate)["status"] == "approved"
    copied = tmp_path / "relocated.sqlite"
    import shutil
    shutil.copy2(first.path, copied)
    builder.validate_activity_provenance_store(target, copied)
