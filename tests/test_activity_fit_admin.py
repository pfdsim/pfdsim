"""Protected root signup, durable review, builder publication and recovery."""

from copy import deepcopy
from contextlib import contextmanager
import json

import numpy as np
import pytest

import app as web
from activity_fit_store import ActivityFitStore
from web_jobs import JobStore
from scripts import build_cas_interaction_parameters as builder
from thermodynamics_models.interaction_fitting import (
    fit_interactions,
    prepare_fit,
    validate_runtime_inclusion,
)
from .test_interaction_fitting import request, synthetic


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


def test_manual_page_preview_and_publication_are_root_only(clients):
    root, guest = clients
    data = {"parameters": {"components": ["ethanol", "water"], "model": "NRTL", "basis": "energy", "values": {"12": 900, "21": -200}, "fit_method": "Source least squares", "statistics": "RMS 1.2%"}, "source": {"citation": "Raw parameter source"}}
    assert guest.get("/parameter-publishing").status_code == 403
    assert guest.post("/api/fitting/admin/preview", json=data).status_code == 403
    assert guest.post("/api/fitting/admin/publish", json=data).status_code == 403
    root_signup(root)
    assert root.post("/api/fitting/admin/publish", json={**data, "id": "unrelated"}).status_code == 400
    page = root.get("/parameter-publishing")
    assert page.status_code == 200
    assert b'id="fit-admin"' in page.data
    assert b'id="publish-statistics"' in page.data
    assert b'id="fit-admin"' not in root.get("/parameter-fitting").data
    preview = root.post("/api/fitting/admin/preview", json={**data, "kind": "GAMMA"})
    assert preview.status_code == 202
    assert web.jobs().get(preview.get_json()["job_id"])["kind"] == "fit_parameter_preview"
    assert web.activity_fit_store().list() == []
    invalid = root.post("/api/fitting/admin/publish", json={**data, "source": {"citation": ""}})
    assert invalid.status_code == 400
    assert web.activity_fit_store().list() == []
    published = root.post("/api/fitting/admin/publish", json=data)
    assert published.status_code == 202, published.get_json()
    job = web.jobs().get(published.get_json()["job_id"])
    assert job["kind"] == "fit_publish"
    stored = web.activity_fit_store().get(web.activity_fit_store().list()[0]["id"])
    assert stored["status"] == "approved"
    assert stored["result"]["reported_fit"] == {"method": "Source least squares", "statistics": "RMS 1.2%"}
    assert stored["events"][0]["details"]["origin"] == "manual_parameters"
    assert root.post("/api/fitting/export", json={"result": stored["result"]}).status_code == 200
    assert root.post("/api/fitting/admin/publish", json=data).status_code == 400
    ordinary = web.jobs().register("other-admin", "long administrator password", "guest:other")
    with web.jobs().connect() as db:
        db.execute("INSERT INTO account_roles VALUES (?, 'admin')", (ordinary["id"],))
    with guest.session_transaction() as session:
        session["user_id"] = ordinary["id"]
    assert guest.get("/parameter-publishing").status_code == 403
    assert guest.post("/api/fitting/admin/preview", json=data).status_code == 403
    assert guest.post("/api/fitting/admin/publish", json=data).status_code == 403


def test_manual_publication_uses_the_shared_builder(tmp_path, monkeypatch):
    from thermodynamics_models.manual_parameters import prepare_manual_parameters

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("PFDSIM_INTERACTION_DATA_DIR", str(runtime))
    store = ActivityFitStore(tmp_path / "manual.sqlite")
    result, _ = prepare_manual_parameters({"components": ["ethanol", "water"], "model": "UNIQUAC", "basis": "tau", "values": {"12": .6, "21": 1.2}, "fit_method": "Published regression", "statistics": "AAD 2%"})
    identifier = store.submit("user:root", "manual", {"citation": "Manual source"}, result)["id"]
    store.review(identifier, "user:root", "approve", None, expected_version=1)
    first = builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root")
    payload = json.loads((runtime / first["runtime_file"]).read_text())
    record = next(item for item in payload["interactions"] if item.get("user_fit_id") == identifier)
    assert record["tau12_a"] == result["parameters"]["tau12_a"]
    assert record["fit_status"] == "admin_published_manual_parameters"
    assert record["fit_provenance"]["reported_fit"]["statistics"] == "AAD 2%"
    second = builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root")
    assert (store.path.parent / second["backup"]).is_file()
    withdrawn = builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root", action="withdraw", notes="Wrong convention")
    assert withdrawn["status"] == "revoked"
    assert not any(item.get("user_fit_id") == identifier for item in json.loads((runtime / first["runtime_file"]).read_text())["interactions"])


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


def test_candidate_publication_and_full_rebuild_export_identical_order(tmp_path, fitted):
    store = ActivityFitStore(tmp_path/'user_activity_fits.sqlite')
    first = store.submit('user:root', 'ethanol', {'citation': 'First source'}, fitted)['id']
    store.review(first, 'user:root', 'approve', 'Reviewed', expected_version=1)
    store.publication_state(first, 'user:root', 'publishing', {})
    store.publication_state(first, 'user:root', 'published', {})

    observations, _, _ = synthetic(
        settings={'components': ['methanol', 'water']}, kinds=('GAMMA_INF', 'HE'),
    )
    second_fit = fit_interactions(observations)
    second = store.submit('user:root', 'methanol', {'citation': 'Second source'}, second_fit)['id']
    store.review(second, 'user:root', 'approve', 'Reviewed', expected_version=1)
    base = {'metadata': {}, 'interactions': []}
    candidate = builder.apply_user_activity_overlay(
        deepcopy(base), 'NRTL', user_fits_path=store.path, candidate=second,
    )
    store.publication_state(second, 'user:root', 'publishing', {})
    store.publication_state(second, 'user:root', 'published', {})
    rebuilt = builder.apply_user_activity_overlay(deepcopy(base), 'NRTL', user_fits_path=store.path)
    assert candidate == rebuilt
    assert {record['user_fit_id'] for record in rebuilt['interactions']} == {first, second}
    assert all('fit_provenance' in record for record in rebuilt['interactions'])


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


def test_publication_allows_warnings_and_failed_phase_assessment(tmp_path, fitted, monkeypatch):
    edited = deepcopy(fitted)
    edited["success"] = False
    edited["optimizer"]["success"] = False
    edited["warnings"] = [
        "The optimizer stopped without convergence.",
        "Parameters at bounds: 12.linear",
        "One or more equilibrium observations failed the final equilibrium/phase-stability audit.",
    ]
    edited["request"]["observations"].append({
        "id":"audit-lle", "kind":"LLE", "T_K":325, "x1_alpha":.1, "x1_beta":.9,
    })
    validation = validate_runtime_inclusion(edited)
    assert any("audit-lle" in warning for warning in validation["warnings"])
    assert edited["warnings"] == validation["warnings"][:3]
    store = ActivityFitStore(tmp_path / "warning-fits.sqlite")
    identifier = store.submit("user:root","warning-fit",{"citation":"Reviewed fit with warnings"},edited)["id"]
    store.review(identifier,"user:root","approve","Warnings reviewed",expected_version=1)
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(builder,"DATA",runtime)
    monkeypatch.setattr(builder,"resolve_component_ids",lambda:([],[]))
    monkeypatch.setattr(builder,"build_interaction_payload",lambda source,resolved,unresolved,**settings:
        builder.apply_user_activity_overlay(
            {"metadata":{},"interactions":[]},"NRTL",user_fits_path=settings["user_fits_path"],
            candidate=settings["user_fit_candidate"],exclude=settings["exclude_user_fit"],
        )
    )
    result = builder.publish_user_fit(identifier,user_fits_path=store.path,actor="user:root")
    assert result["status"] == "published"
    payload = json.loads((runtime / result["runtime_file"]).read_text())
    assert any(record.get("user_fit_id") == identifier for record in payload["interactions"])
    saved = store.get(identifier)
    assert saved["result"]["success"] is False
    assert saved["result"]["warnings"] == edited["warnings"]
    event = next(event for event in saved["events"] if event["action"] == "publishing")
    assert any("audit-lle" in warning for warning in event["details"]["validation"]["warnings"])


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
@pytest.mark.parametrize("basis", ["pressure_correction", "subcooled_liquid"])
def test_publication_accepts_reviewed_psat_basis_without_changing_it(
    tmp_path, monkeypatch, model, basis,
):
    if basis == "pressure_correction":
        components = ["acetone", "water"]
        psat = [{
            "form": "antoine",
            "coefficients": {"A": 4.296305, "B": 1230.4062, "C": -43.24334},
            "temperature_unit": "K", "pressure_unit": "bar",
            "Tmin_K": 273, "Tmax_K": 400,
            "source": "Source pressure correction",
        }, None]
        temperatures = (300, 350)
    else:
        components = ["acetone", "cyclohexane"]
        psat = [None, {
            "form": "canonical_psat_af",
            "coefficients": {
                "A": 248.67589730698694, "B": -9896.656129531802,
                "C": -43.13132713304277, "D": 0.11134395321363802,
                "E": -5.645824565305427e-05, "F": 1.9410217270456763e-14,
            },
            "temperature_unit": "K", "pressure_unit": "bar",
            "Tmin_K": 273, "Tmax_K": 400,
            "source": "Subcooled liquid Psat range extension",
        }]
        temperatures = (273.15, 300)
    data = request(components=components, model=model, psat=psat)
    original = prepare_fit(data)
    values = np.array([0.2, 0.3, 0.1, 0.2])
    original.install(values)
    data["initial"] = dict(zip(original.names, values * original.scales))
    data["observations"] = []
    for T in temperatures:
        for x in (0.2, 0.5, 0.8):
            point = original.predict_vle(x, T=T)
            data["observations"].append({
                "kind": "VLE", "T_K": T, "x1": x,
                "P_bar": point["P_bar"], "y1": point["y1"],
            })
    data["observations"][0]["pin"] = True
    result = fit_interactions(data)
    assert result["success"]
    assert result["points"][0]["pin_satisfied"]
    assert validate_runtime_inclusion(result)["property_basis"] == "shared_liquid_activity_verified"

    # Exercise the real publication builder in isolation, retaining custom Psat
    # in provenance while publishing only the unchanged liquid coefficients.
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("PFDSIM_INTERACTION_DATA_DIR", str(runtime))
    store = ActivityFitStore(tmp_path / "source" / "fits.sqlite")
    identifier = store.submit("user:root", "job", {"citation": "Reviewed source"}, result)["id"]
    store.review(identifier, "user:root", "approve", "Reviewed Psat", expected_version=1)
    published = builder.publish_user_fit(identifier, user_fits_path=store.path, actor="user:root")
    assert published["status"] == "published"
    record = next(
        item for item in json.loads((runtime / published["runtime_file"]).read_text())["interactions"]
        if item.get("user_fit_id") == identifier
    )
    for field, value in result["parameters"].items():
        assert record[field] == value
    assert store.get(identifier)["result"]["request"]["psat"] == result["request"]["psat"]
    assert result["request"]["psat"] == psat


def test_publication_still_recomputes_hard_pins_on_original_basis(fitted):
    edited = deepcopy(fitted)
    row = next(row for row in edited["request"]["observations"] if row["kind"] == "GAMMA_INF")
    row["gamma1_inf"] *= 2
    row["pin"] = True
    with pytest.raises(ValueError, match="hard-pin constraints"):
        validate_runtime_inclusion(edited)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_publication_warning_policy_still_rejects_nonfinite_coefficients(fitted, value):
    edited = deepcopy(fitted)
    edited["success"] = False
    edited["warnings"] = ["Fit needs review"]
    edited["coefficients"]["12.constant"] = value
    with pytest.raises(ValueError, match="finite"):
        validate_runtime_inclusion(edited)


def test_unavailable_measurement_audit_is_a_publication_warning(fitted, monkeypatch):
    from thermodynamics_models.interaction_fitting import FitProblem

    def unavailable(*args, **kwargs):
        raise ValueError("Measurement audit unavailable")

    monkeypatch.setattr(FitProblem, "report", unavailable)
    edited = deepcopy(fitted)
    edited["success"] = False
    verification = validate_runtime_inclusion(edited)
    assert any("Measurement audit unavailable" in warning for warning in verification["warnings"])


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
