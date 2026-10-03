"""HTTP contracts, lossless model editing, and persistent calculation lifecycle."""

from collections import OrderedDict
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytest.importorskip("flask")
import app as web
from pfd_parser import Metadata, parse_pfd
from unit_syntax import UNIT_TYPE_ALIASES, port_schema_for_unit_type
from web_jobs import JobStore, calculate_chart, calculate_simulation, finite_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(web.app.config, "TESTING", True)
    monkeypatch.setitem(web.app.config, "JOB_DIRECTORY", tmp_path)
    client = web.app.test_client()
    token = client.get("/api/session").get_json()["csrf_token"]
    client.environ_base["HTTP_X_CSRF_TOKEN"] = token
    return client


@pytest.mark.parametrize("path", ["/", "/editor", "/settings", "/vle-chart"])
def test_screens_use_local_assets(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"/static/css/laboratory.css" in response.data
    assert b"cdn.jsdelivr" not in response.data
    assert b"fonts.googleapis" not in response.data


def test_templates_follow_current_registry(client):
    templates = client.get("/api/unit-templates").get_json()
    assert set(templates) == set(UNIT_TYPE_ALIASES.values())
    for name, template in templates.items():
        schema = port_schema_for_unit_type(name)
        assert template["ports"]
        text = f"UNIT U : {name}\n"
        pfd = parse_pfd(text).to_dict()
        pfd["units"][0]["ports"] = template["ports"]
        response = client.post("/api/serialize", json={"pfd": pfd})
        assert response.status_code == 200, (name, response.get_json())
        assert all(
            port["port_type"] in set(schema["types"].values())
            for port in template["ports"]
        )


def test_metadata_json_preserves_all_fields():
    pfd = parse_pfd(
        "PROCESS: Round trip\nALLOW_COMPUTATION: false\nONLINE_LOOKUP: false\n"
    )
    assert set(pfd.to_dict()["metadata"]) == {field.name for field in fields(Metadata)}
    restored = type(pfd).from_dict(pfd.to_dict())
    assert restored.metadata.allow_computation is False
    assert "ALLOW_COMPUTATION: false" in restored.to_pfd()


def test_reaction_controls_follow_registered_runtime_capabilities(client):
    templates = client.get("/api/unit-templates").get_json()
    assert {
        name for name, template in templates.items() if template["supports_reactions"]
    } == {
        "Reactor",
        "EquilibriumReactor",
        "CSTR",
        "PFR",
        "PackedBedReactor",
        "BatchReactor",
    }
    assert all(
        isinstance(template["supports_reactions"], bool)
        for template in templates.values()
    )


@pytest.mark.parametrize(
    "filename",
    [
        "simple_flash.pfd",
        "permanent_solid_slurry_operations.pfd",
        "cake_filtration_washing.pfd",
        "jacketed_cstr_ignition_extinction.pfd",
        "isopropanol_water_nrtl_uniquac_comparison.pfd",
        "equilibrium_warm_melt_washing.pfd",
        "dcm_3a_molecular_sieve_drying.pfd",
        "ethanol_pressure_swing_recycle_wasteful.pfd",
    ],
)
def test_advanced_example_editing_preserves_model(client, filename):
    original = parse_pfd((ROOT / "examples" / filename).read_text())
    data = original.to_dict()
    for i, unit in enumerate(data["units"]):
        unit["x"], unit["y"] = i * 321, i * 177
    response = client.post("/api/serialize", json={"pfd": data})
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert parse_pfd(result["text"]).to_dict() == original.to_dict()
    assert [(u["x"], u["y"]) for u in result["pfd"]["units"]] == [
        (u["x"], u["y"]) for u in data["units"]
    ]


def test_serialize_rejects_invalid_edit_without_emitting_bad_text(client):
    pfd = parse_pfd("UNIT U : Pump\n").to_dict()
    pfd["units"][0]["unit_type"] = "GibbsReactor"
    response = client.post("/api/serialize", json={"pfd": pfd})
    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert "text" not in response.get_json()


@pytest.mark.parametrize("payload", [None, [], {"text": ""}, {"text": 1}])
def test_parse_rejects_bad_request_shapes(client, payload):
    response = client.post("/api/parse", json=payload)
    assert response.status_code in {400, 415}
    assert "traceback" not in response.get_json()


def test_invalid_simulation_controls_never_start_job(client, monkeypatch):
    monkeypatch.setattr(
        web, "submit", lambda *args: pytest.fail("Invalid input started a job")
    )
    for controls in [
        {"max_iterations": 1.5},
        {"max_iterations": True},
        {"tolerance": 0},
        {"tolerance": None},
    ]:
        response = client.post(
            "/api/simulate", json={"text": "PROCESS: Valid text\n", **controls}
        )
        assert response.status_code == 400


def test_same_canonical_flowsheet_has_same_affinity(client, monkeypatch):
    submitted = []
    monkeypatch.setattr(
        web,
        "submit",
        lambda kind, payload: (
            submitted.append(payload) or web.respond({"success": True}, 202)
        ),
    )
    for text in [
        "PROCESS: Same\n# first comment\n",
        "PROCESS: Same\n# changed comment\n",
    ]:
        assert client.post("/api/simulate", json={"text": text}).status_code == 202
    assert submitted[0]["fingerprint"] == submitted[1]["fingerprint"]
    client.post("/api/simulate", json={"text": "PROCESS: Changed\n"})
    assert submitted[-1]["fingerprint"] != submitted[0]["fingerprint"]


def test_pfr_download_uses_completed_output(client, monkeypatch):
    store = JobStore(web.app.config["JOB_DIRECTORY"])
    with store.connect() as db:
        with client.session_transaction() as session:
            owner = "guest:" + session["guest_id"]
        db.execute(
            "INSERT INTO jobs (id,kind,status,created,updated,payload,output,progress,slot,owner) VALUES ('done','simulation','completed',?,?,?,?,'[]',0,?)",
            (
                time.time(),
                time.time(),
                "{}",
                json.dumps({"pfr_content": "existing results"}),
                owner,
            ),
        )
    monkeypatch.setattr(
        JobStore, "submit", lambda *args: pytest.fail("Download reran simulation")
    )
    response = client.post(
        "/api/simulate/download",
        json={"job_id": "done", "filename": "../../result.pfr"},
    )
    assert response.status_code == 200
    assert response.data == b"existing results"
    assert ".." not in response.headers["Content-Disposition"]
    assert (
        client.post(
            "/api/simulate/download", json={"text": "PROCESS: Another\n"}
        ).status_code
        == 400
    )


def test_json_sanitizes_unavailable_numerical_results():
    assert finite_json({"array": [float("nan"), float("inf"), 3]}) == {
        "array": [None, None, 3]
    }


def test_local_chart_payload_validation(client, monkeypatch):
    payloads = []
    monkeypatch.setattr(
        web,
        "submit",
        lambda kind, payload: (
            payloads.append(payload) or web.respond({"success": True}, 202)
        ),
    )
    request = {
        "comp1": "water",
        "comp2": "ethanol",
        "method": "NRTL",
        "chart_type": "PXY",
        "online_lookup": False,
    }
    assert client.post("/api/vle-chart", json=request).status_code == 202
    assert parse_pfd(payloads[0]["text"]).metadata.online_lookup is False
    assert (
        client.post(
            "/api/vle-chart",
            json={
                key: value for key, value in request.items() if key != "online_lookup"
            },
        ).status_code
        == 202
    )
    assert parse_pfd(payloads[1]["text"]).metadata.online_lookup is True
    for invalid in [
        {"n_points": 201},
        {"temperature": -274},
        {"pressure": 0},
        {"online_lookup": "false"},
        {"chart_type": "invalid"},
        {"method": "STEAM"},
    ]:
        assert (
            client.post("/api/vle-chart", json={**request, **invalid}).status_code
            == 400
        )


def test_chart_imports_definitions_without_requiring_a_finished_process(
    client, monkeypatch
):
    submitted = []
    monkeypatch.setattr(
        web,
        "submit",
        lambda kind, payload: (
            submitted.append(payload) or web.respond({"success": True}, 202)
        ),
    )
    text = """PROCESS: Unfinished process
THERMO_METHOD: NRTL
FLUID_PHASE_MODEL: VLLE
ONLINE_LOOKUP: false
COMPONENTS:
    A | ethanol
    B | water
STREAM Feed : FEED -> U.in
UNIT U : Pump
"""
    response = client.post(
        "/api/vle-chart",
        json={
            "text": text,
            "comp1": "A",
            "comp2": "B",
            "method": "IDEAL",
            "chart_type": "PXY",
        },
    )
    assert response.status_code == 202, response.get_json()
    definition = parse_pfd(submitted[0]["text"])
    assert not definition.units and not definition.streams
    assert definition.metadata.online_lookup is False
    result = calculate_chart(submitted[0], lambda message: None, OrderedDict())
    assert result["comp1"] == "A" and result["comp2"] == "B"
    assert not result["errors"]


def test_scoped_chart_options_cannot_be_silently_ignored(client, monkeypatch):
    monkeypatch.setattr(
        web, "submit", lambda *args: pytest.fail("Unsupported options started a job")
    )
    response = client.post(
        "/api/vle-chart",
        json={
            "comp1": "water",
            "comp2": "ethanol",
            "method": "NRTL-BV",
            "scope": "local",
            "text": "THERMO_SCOPES:\n    local | method=NRTL-BV\n",
            "thermo_options": {"correlation": "HOC"},
        },
    )
    assert response.status_code == 400
    assert "global scope" in response.get_json()["error"]


def test_phase_inputs_are_plain_text_and_reference_list_is_optional(client):
    page = client.get("/vle-chart").get_data(as_text=True)
    assert 'list="chemical-options"' not in page
    assert "<datalist" not in page
    assert "reference-browser" in page
    response = client.get("/api/vle-chart/components").get_json()
    assert any("Perry" in item["source"] for item in response["components"])
    assert any("chemicals.json" in item["source"] for item in response["components"])
    assert len(response["components"]) > len(web.get_database().list_all())


def test_user_layout_positions_and_routes_remain_consistent(client):
    pfd = parse_pfd((ROOT / "examples/simple_flash.pfd").read_text()).to_dict()
    pfd["units"][0]["x"], pfd["units"][0]["y"] = 500, 320
    pfd["units"][1]["x"], pfd["units"][1]["y"] = 1100, 500
    response = client.post("/api/layout", json={"pfd": pfd, "keep_positions": True})
    assert response.status_code == 200, response.get_json()
    layout = response.get_json()["layout"]
    assert abs(layout["units"]["HEAT-1"]["x"] - 500) < 1e-6
    assert abs(layout["units"]["HEAT-1"]["y"] - 320) < 1e-6
    for stream in pfd["streams"]:
        route = layout["streams"][stream["id"]]["points"]
        for ref, point in [
            (stream["source"], route[0]),
            (stream["destination"], route[-1]),
        ]:
            if ref["unit_id"]:
                unit = layout["units"][ref["unit_id"]]
                port = unit["ports"][ref["port_id"]]
                assert abs(point[0] - unit["x"] - port["x"]) < 0.002
                assert abs(point[1] - unit["y"] - port["y"]) < 0.002


def test_unfinished_feed_does_not_block_drawing(client):
    pfd = parse_pfd("STREAM Feed : FEED -> U.in\nUNIT U : Pump\n").to_dict()
    response = client.post("/api/layout", json={"pfd": pfd})
    assert response.status_code == 200, response.get_json()
    assert "Feed" in response.get_json()["layout"]["streams"]


def test_actual_solver_reuses_only_same_flowsheet():
    canonical = parse_pfd((ROOT / "examples/simple_flash.pfd").read_text()).to_pfd()
    payload = {
        "text": canonical,
        "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "max_iterations": 100,
        "tolerance": 1e-4,
    }
    simulators = OrderedDict()
    first = calculate_simulation(payload, lambda message: None, simulators)
    initialized = next(iter(simulators.values())).solver
    second = calculate_simulation(payload, lambda message: None, simulators)
    assert first["converged"] and second["converged"]
    assert second["reused_flowsheet"]
    assert next(iter(simulators.values())).solver is initialized
    assert first["results"]["streams"] == second["results"]["streams"]
    changed = canonical.replace("T_out = 80", "T_out = 75")
    third = calculate_simulation(
        {
            **payload,
            "text": changed,
            "fingerprint": hashlib.sha256(changed.encode()).hexdigest(),
        },
        lambda message: None,
        simulators,
    )
    assert not third["reused_flowsheet"]
    assert len(simulators) == 2


def test_phase_atlas_uses_pfd_custom_symbols_properties_and_scope():
    pfd = parse_pfd((ROOT / "examples/simple_flash.pfd").read_text())
    pfd.metadata.online_lookup = False
    payload = {
        **web.flowsheet_payload(pfd),
        "components": ["C2H5OH", "H2O"],
        "scope": "global",
        "chart_type": "PXY",
        "method": "IDEAL",
        "temperature": 50,
        "pressure": 1,
        "n_points": 4,
    }
    simulators = OrderedDict()
    result = calculate_chart(payload, lambda message: None, simulators)
    assert result["comp1"] == "C2H5OH"
    assert result["comp2"] == "H2O"
    assert result["method"] == "IDEAL"
    assert not result["reused_flowsheet"]
    assert calculate_chart(payload, lambda message: None, simulators)["reused_flowsheet"]


def test_queue_admission_is_bounded_and_cancellation_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path, capacity=1)
    identifiers = [store.submit("chart", {"x": i}) for i in range(4)]
    with pytest.raises(RuntimeError, match="queue is full"):
        store.submit("chart", {})
    assert store.cancel(identifiers[0])["status"] == "cancelled"
    store.update(identifiers[0], status="completed", output="{}")
    assert store.get(identifiers[0])["status"] == "cancelled"
    assert store.submit("chart", {})


def test_persistent_worker_cancellation_stops_its_calculation(tmp_path, monkeypatch):
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    store = JobStore(tmp_path, capacity=1)
    identifier = store.submit("simulation", {})
    token = "test-worker"
    with store.connect() as db:
        db.execute("INSERT INTO workers VALUES (0,?,NULL,?)", (token, time.time()))
    code = """import time,sys
import web_jobs
web_jobs.calculate_simulation = lambda *args: time.sleep(60)
web_jobs.run_worker(web_jobs.JobStore(sys.argv[1], capacity=1), 0, 'test-worker')
"""
    with (tmp_path / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path)],
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
    try:
        deadline = time.monotonic() + 10
        while (
            store.get(identifier)["status"] != "running" and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert store.get(identifier)["status"] == "running"
        store.cancel(identifier)
        assert child.wait(timeout=5) == 0
        assert store.get(identifier)["status"] == "cancelled"
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_reused_pid_does_not_leave_running_job_or_worker_slot_stuck(
    tmp_path, monkeypatch
):
    import web_jobs

    store = JobStore(tmp_path, capacity=1)
    with store.connect() as db:
        db.execute(
            "INSERT INTO workers VALUES (0,'old-worker',?,?)",
            (os.getpid(), time.time() - 30),
        )
        db.execute(
            """INSERT INTO jobs (id,kind,status,created,updated,payload,progress,slot,pid)
            VALUES ('orphan','simulation','running',?,?,?,'[]',0,?)""",
            (time.time() - 30, time.time() - 30, "{}", os.getpid()),
        )
    assert store.get("orphan")["status"] == "failed"
    started = []

    class Child:
        pid = os.getpid()

        def wait(self):
            return 0

    monkeypatch.setattr(
        web_jobs.subprocess,
        "Popen",
        lambda command, **kwargs: started.append(command) or Child(),
    )
    store.ensure_worker(0)
    assert len(started) == 1
    assert "--token" in started[0]


def test_parse_auto_layout_is_deterministic_and_handles_cycles(client):
    text = "STREAM A : U.out -> V.in\nSTREAM B : V.out -> U.in\nUNIT U : Pump\nUNIT V : Pump\n"
    first = client.post("/api/parse", json={"text": text}).get_json()
    second = client.post("/api/parse", json={"text": text}).get_json()
    assert first["pfd"] == second["pfd"]
    assert first["pfd"]["units"][0]["x"] != first["pfd"]["units"][1]["x"]


def test_api_errors_are_json_and_uncached(client):
    response = client.get("/api/jobs/not-a-job")
    assert response.status_code == 404
    assert response.is_json
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
