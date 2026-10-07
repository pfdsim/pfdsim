"""Fitting HTTP contracts, owned exports and durable sourced submissions."""

import json

import pytest

import app as web
from cli import main as cli_main
from .test_interaction_fitting import request, synthetic
from thermodynamics_models.interaction_fitting import fit_interactions
from web_jobs import JobStore, calculate_fit, calculate_fit_prefill


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(web.app.config, "TESTING", True)
    monkeypatch.setitem(web.app.config, "JOB_DIRECTORY", tmp_path)
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    client = web.app.test_client()
    token = client.get("/api/session").get_json()["csrf_token"]
    client.environ_base["HTTP_X_CSRF_TOKEN"] = token
    return client


def completed(client, result):
    response = client.post("/api/fitting", json=result["request"])
    assert response.status_code == 202
    identifier = response.get_json()["job_id"]
    web.jobs().update(identifier, status="completed", output=json.dumps(result))
    return identifier


@pytest.fixture(scope="module")
def fitted():
    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    return fit_interactions(data)


def test_page_catalog_parser_and_admission(client):
    response = client.get("/parameter-fitting")
    assert response.status_code == 200
    assert b"js/parameter-fitting.js" in response.data
    assert client.get("/api/fitting/catalog").get_json()["kinds"] == [
        "VLE",
        "LLE",
        "HE",
        "GAMMA_INF",
        "AZEOTROPE",
        "VLLE",
        "UCST",
        "LCST",
    ]
    response = client.post(
        "/api/fitting/parse",
        json={"observations": "kind,T_C,x1,HE_J_mol\nHE,25,0.5,200"},
    )
    assert response.get_json()["observations"][0]["T_K"] == 298.15
    assert (
        client.post("/api/fitting", json=request(weights={"unknown": 2})).status_code
        == 400
    )


def test_missing_values_corrections_and_flat_dimensions_http(client):
    response = client.post("/api/fitting/parse", json={
        "observations": [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2, "gamma2_inf": None}],
    })
    assert response.status_code == 200
    assert "gamma2_inf" not in response.get_json()["observations"][0]
    source = "kind,T_K,x1,HE_J_mol\nHE,300,.2,?"
    response = client.post("/api/fitting/parse", json={"observations": source})
    assert response.status_code == 200 and not response.get_json()["ready"]
    response = client.post("/api/fitting/parse", json={"observations": source, "import_options": {
        "cell_edits": [{"row": 0, "column": 3, "value": "-20"}],
    }})
    assert response.status_code == 200 and response.get_json()["ready"]
    assert response.get_json()["observations"][0]["HE_J_mol"] == -20
    source = "300 2 ? 310 None 3"
    response = client.post("/api/fitting/parse", json={"observations": source})
    assert response.status_code == 200 and response.get_json()["dimensions_needed"]
    response = client.post("/api/fitting/parse", json={"observations": source, "import_options": {
        "kind": "GAMMA_INF", "row_count": 2, "column_count": 3, "temperature_unit": "K",
        "mapping": ["temperature", "gamma1_inf", "gamma2_inf"], "layout": "rows",
    }})
    assert response.status_code == 200 and response.get_json()["ready"]
    assert len(response.get_json()["observations"]) == 2


def test_one_sided_lle_import_and_job_admission(client):
    response = client.post("/api/fitting/parse", json={
        "observations": "kind,T_K,x1_alpha,x1_beta\nLLE,300,.1,—\nLLE,310,None,.8",
    })
    assert response.status_code == 200
    proposal = response.get_json()
    assert proposal["ready"], proposal["issues"]
    rows = proposal["observations"]
    assert rows[0]["x1_alpha"] == 0.1 and "x1_beta" not in rows[0]
    assert rows[1]["x1_beta"] == 0.8 and "x1_alpha" not in rows[1]
    assert client.post("/api/fitting", json=request(observations=rows)).status_code == 202
    invalid = request(observations=[{"kind": "LLE", "T_K": 300}])
    assert client.post("/api/fitting", json=invalid).status_code == 400


@pytest.mark.parametrize("axis", ["T", "P"])
def test_grouped_vle_table_http_preserves_conditions_and_ignores_fitted_values(client, axis):
    from .test_grouped_vle_import import grouped_table, assert_grouped_points

    response = client.post("/api/fitting/parse", json={"observations": grouped_table(axis, junk=True)})
    assert response.status_code == 200
    assert_grouped_points(response.get_json(), axis)


def test_worker_uses_shared_fitting_and_prefill():
    data = request(
        model="UNIQUAC",
        form="constant",
        observations=[{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1}],
    )
    prefilled = calculate_fit_prefill(data, lambda message: None)
    assert all(item["r"] > 0 and item["q"] > 0 for item in prefilled["rq"])
    result = calculate_fit(request(form="constant"), lambda message: None)
    assert result["success"]


def test_rq_prefill_ignores_joint_vapor_fitting_settings(client):
    response = client.post("/api/fitting/prefill", json=request(
        vapor="PR", vapor_parameters=[{
            "model": "PR", "field": "kij", "fit": True, "value": 0,
        }],
    ))
    assert response.status_code == 202, response.get_json()
    with web.jobs().connect() as db:
        payload = json.loads(db.execute(
            "SELECT payload FROM jobs WHERE id=?", (response.get_json()["job_id"],)
        ).fetchone()[0])
    output = calculate_fit_prefill(payload, lambda message: None)
    assert all(item["r"] > 0 and item["q"] > 0 for item in output["rq"])


def test_login_preserves_guest_and_account_submissions_of_the_same_report(client, fitted):
    credentials = {"username": "fit-owner", "password": "A long account password"}
    response = client.post("/api/account/register", json=credentials)
    assert response.status_code == 200
    client.environ_base["HTTP_X_CSRF_TOKEN"] = response.get_json()["csrf_token"]
    account_fit = client.post("/api/fitting/submit", json={
        "result": fitted, "source": {"citation": "Account source notes"},
    }).get_json()["submission"]["id"]
    response = client.post("/api/account/logout", json={})
    client.environ_base["HTTP_X_CSRF_TOKEN"] = response.get_json()["csrf_token"]
    guest_fit = client.post("/api/fitting/submit", json={
        "result": fitted, "source": {"citation": "Guest source notes"},
    }).get_json()["submission"]["id"]
    guest_record = web.activity_fit_store().get(guest_fit)
    response = client.post("/api/account/login", json=credentials)
    assert response.status_code == 200, response.get_json()
    submissions = client.get("/api/fitting/submissions").get_json()["submissions"]
    assert {item["id"] for item in submissions} == {account_fit, guest_fit}
    assert {item["source"]["citation"] for item in submissions} == {
        "Account source notes", "Guest source notes",
    }
    claimed = web.activity_fit_store().get(guest_fit)
    assert claimed["owner"] == web.activity_fit_store().get(account_fit)["owner"]
    assert claimed["result"] == guest_record["result"]
    assert claimed["events"][-1]["details"]["previous_job_id"] == guest_record["job_id"]


def test_export_ownership_and_source_required(client, fitted):
    identifier = completed(client, fitted)
    exported = client.post(
        "/api/fitting/export", json={"job_id": identifier}
    ).get_json()
    assert exported["entry"] == fitted["entry"]
    assert (
        client.post(
            "/api/fitting/submit",
            json={"job_id": identifier, "source": {"citation": ""}},
        ).status_code
        == 400
    )
    stranger = web.app.test_client()
    token = stranger.get("/api/session").get_json()["csrf_token"]
    assert (
        stranger.post(
            "/api/fitting/export",
            json={"job_id": identifier},
            headers={"X-CSRF-Token": token},
        ).status_code
        == 400
    )
    source = {
        "citation": "Synthetic validation dataset; not experimental literature",
        "notes": "For testing only",
    }
    response = client.post(
        "/api/fitting/submit", json={"job_id": identifier, "source": source}
    )
    assert response.status_code == 201
    submission = response.get_json()["submission"]
    repeated = client.post(
        "/api/fitting/submit", json={"job_id": identifier, "source": source}
    ).get_json()
    assert repeated["submission"]["id"] == submission["id"]
    assert submission["status"] == "pending_review"
    # Retention is independent of the temporary job's lifecycle.
    with web.jobs().connect() as db:
        db.execute("DELETE FROM jobs WHERE id=?", (identifier,))
    stored = web.activity_fit_store().get(submission["id"])["result"]
    assert stored["parameters"] == fitted["parameters"]
    assert (
        client.get("/api/fitting/submissions").get_json()["submissions"][0]["source"]
        == source
    )
    assert stranger.get("/api/fitting/submissions").get_json()["submissions"] == []


def test_cli_report_can_be_submitted_for_review(client, fitted):
    response = client.post(
        "/api/fitting/submit",
        json={
            "result": fitted,
            "source": {"citation": "Source for externally computed fit"},
        },
    )
    assert response.status_code == 201
    stored = web.activity_fit_store().get(response.get_json()["submission"]["id"])[
        "result"
    ]
    assert stored["submission_origin"] == "external_report_pending_review"
    assert (
        client.post(
            "/api/fitting/submit",
            json={
                "result": {**fitted, "success": False},
                "source": {"citation": "Source"},
            },
        ).status_code
        == 201
    )


def test_review_cli_reads_persisted_bundle(client, fitted, tmp_path):
    identifier = completed(client, fitted)
    response = client.post(
        "/api/fitting/submit",
        json={"job_id": identifier, "source": {"citation": "Review source"}},
    )
    submission = response.get_json()["submission"]["id"]
    output = tmp_path / "review.json"
    assert (
        cli_main(
            [
                "fit",
                "submissions",
                "--directory",
                str(tmp_path),
                "--id",
                submission,
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert (
        json.loads(output.read_text())["result"]["parameters"] == fitted["parameters"]
    )


def test_archived_report_export_does_not_require_live_job(client, fitted):
    response = client.post("/api/fitting/export", json={"result": fitted})
    assert response.status_code == 200
    from pfd_parser import parse_pfd

    assert (
        parse_pfd(response.get_json()["pfd_text"]).to_dict()
        == parse_pfd(fitted["pfd_text"]).to_dict()
    )


def test_multiline_submission_citation_is_preserved_outside_model_parameters(
    client, fitted
):
    source = {
        "citation": 'Authors, "A paper"\nThermochimica Acta, tables 1–3',
        "notes": "Original formatting retained",
    }
    response = client.post(
        "/api/fitting/submit", json={"result": fitted, "source": source}
    )
    assert response.status_code == 201, response.get_json()
    stored = web.activity_fit_store().get(response.get_json()["submission"]["id"])
    assert stored["source"] == source
    assert stored["result"]["parameters"] == fitted["parameters"]
