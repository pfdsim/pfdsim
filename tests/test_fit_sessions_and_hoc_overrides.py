"""Complete saved-state ownership/versioning and HOC override/default handling."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import app as web
from simulator import Simulator
from thermodynamics_models.interaction_fitting import prepare_fit
from thermodynamics_models.hayden_oconnell import hoc_association_group
from web_jobs import JobStore


def hoc_request(eta=None):
    result = {
        "components": ["water", "acetic acid"],
        "model": "NRTL",
        "vapor": "HOC",
        "form": "constant",
        "observations": [{"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.5}],
    }
    if eta is not None:
        result["component_properties"] = [{}, {"hoc_eta": eta}]
    return result


def test_supplied_hoc_eta_is_not_filtered_out_by_liquid_initialization():
    problem = prepare_fit(hoc_request(2.75))
    acid = problem.components[1]
    assert problem.thermo.props[acid].hoc_eta == 2.75
    assert problem.definition.get_component(acid).hoc_eta == 2.75
    record = next(
        item
        for item in problem.property_records
        if item["property"] == "hoc_eta" and item["component"] == acid
    )
    assert record["value"] == 2.75 and record["method"] == "pfd_component_override"
    assert not any(
        "Ignoring HOC pure eta" in message or "Acetic Acid: HOC eta=0" in message
        for message in problem.property_warnings
    )
    assert problem.thermo.vapor_eos.provider.pure_eta_overrides[acid] == 2.75


def test_acetic_acid_uses_recorded_four_point_five_group_default():
    problem = prepare_fit(hoc_request())
    acid = problem.components[1]
    assert problem.thermo.props[acid].hoc_eta == 4.5
    assert hoc_association_group(acid, problem.thermo.props[acid]) == "organic acid"
    assert any(
        "4.5" in message and "recorded organic acid" in message
        for message in problem.property_warnings
    )
    assert not any(
        "Acetic Acid: HOC eta=0" in message for message in problem.property_warnings
    )


def test_acid_user_zero_is_explicit_not_a_missing_value_default():
    problem = prepare_fit(hoc_request(0))
    assert problem.thermo.props[problem.components[1]].hoc_eta == 0
    assert not any(
        "Acetic Acid: HOC eta=0 is the zero default" in warning
        for warning in problem.property_warnings
    )


def test_vdm_component_definition_survives_property_only_initialization():
    data = {
        "components": ["W", "A"],
        "model": "NRTL",
        "vapor": "VDM",
        "form": "constant",
        "pfd_text": "ONLINE_LOOKUP: false\nCOMPONENTS:\n    W | water\n    A | acetic acid | VDM={delta_H=-40000, delta_S=-100}\n",
        "observations": [{"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.5}],
    }
    problem = prepare_fit(data)
    assert problem.thermo.props["A"].vapor_dimerization == {
        "delta_H_J_per_mol": -40000.0,
        "delta_S_J_per_mol_K": -100.0,
    }
    assert not any("Ignoring VDM" in warning for warning in problem.property_warnings)


def test_multifunctional_acid_cannot_silently_receive_zero_eta():
    from thermodynamics_models.fitting_properties import prepare_auxiliary_properties
    from pfd_parser import Component, ProcessFlowDiagram
    from chemical_properties import ChemicalDatabase

    props = SimpleNamespace(
        CAS="",
        name="Multifunctional acid test",
        smiles="O=C(O)CC(=O)O",
        Tc=600,
        Pc=40,
        dipole_moment=1.5,
        modified_radius_of_gyration=2.2,
        hoc_eta=None,
        property_sources={},
    )
    thermo = SimpleNamespace(
        components=["A"],
        props={"A": props},
        db=ChemicalDatabase(enable_online=False),
        _resolver_known_props={
            "A": {
                "name": props.name,
                "smiles": props.smiles,
                "Tc": 600,
                "Pc": 40,
                "dipole_moment": 1.5,
                "modified_radius_of_gyration": 2.2,
            }
        },
    )
    definition = ProcessFlowDiagram(components=[Component("A", props.name)])
    with pytest.raises(ValueError, match="cannot default to zero"):
        prepare_auxiliary_properties(
            thermo,
            definition,
            {
                "vapor": "HOC",
                "online_lookup": False,
                "estimate_properties": True,
                "allow_hoc_eta_default": True,
            },
        )


def test_classifier_exposes_acid_presence_for_unsupported_multifunctional_cases():
    props = SimpleNamespace(CAS="", smiles="O=C(O)CC(=O)O")
    result = hoc_association_group("malonic acid", props, include_features=True)
    assert result["has_carboxylic_acid"] and result["group"] == "multifunctional"


def test_property_initialization_context_changes_cache_and_keeps_pfd_overrides():
    text = "COMPONENTS:\n    W | water\n    A | acetic acid | HOC_eta=3.25\nTHERMO_METHOD: NRTL\nONLINE_LOOKUP: false\n"
    sim = Simulator.from_string(text)
    sim.initialize()
    assert sim.thermo.props["A"].hoc_eta is None
    sim.initialize(property_methods=["NRTL-HOC"])
    assert sim.thermo.props["A"].hoc_eta == 3.25
    assert not any(
        "Ignoring HOC pure eta" in warning for warning in sim.thermo.warnings
    )


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setitem(web.app.config, "TESTING", True)
    monkeypatch.setitem(web.app.config, "JOB_DIRECTORY", tmp_path)
    monkeypatch.setattr(JobStore, "ensure_worker", lambda *args: None)
    clients = [web.app.test_client(), web.app.test_client()]
    for index, client in enumerate(clients):
        token = client.get("/api/session").get_json()["csrf_token"]
        client.environ_base["HTTP_X_CSRF_TOKEN"] = token
        response = client.post(
            "/api/account/register",
            json={
                "username": f"saved-fit-user-{index}",
                "password": "Long fitting-session password",
            },
        )
        assert response.status_code == 200
        client.environ_base["HTTP_X_CSRF_TOKEN"] = response.get_json()["csrf_token"]
    return clients


def session_document():
    return {
        "type": "pfdsim_fit_session",
        "schema_version": 1,
        "name": "HE at five temperatures",
        "state": {
            "controls": {
                "comp1": "acetone",
                "comp2": "1-propanol",
                "model": "UNIQUAC",
                "law": "full",
                "vapor": "HOC",
                "estimate-properties": True,
            },
            "weights": {"HE": 2.5},
            "sigmaValues": {"HE_J_mol": "20"},
            "psatValues": [{}, {"hoc_eta": "4.5"}],
            "observations": [
                {
                    "id": "1",
                    "kind": "HE",
                    "T_K": 300,
                    "x1": 0.5,
                    "HE_J_mol": 100,
                    "sigma": {"HE_J_mol": 20},
                    "pin": False,
                    "validation_only": True,
                }
            ],
            "pendingImport": {
                "text": "Raw paper table",
                "options": {
                    "series": [{"columns": [1], "temperature": 300}],
                    "shared_columns": [0],
                },
            },
            "importReports": [
                {
                    "original_text": "Raw paper table",
                    "observation_sources": [{"row": 0, "series": 0}],
                }
            ],
            "result": None,
        },
    }


def test_saved_sessions_preserve_complete_state_and_are_private_and_versioned(clients):
    first, second = clients
    doc = session_document()
    response = first.post(
        "/api/fitting/sessions/fitting-session-001",
        json={"version": 0, "document": doc},
    )
    assert response.status_code == 200
    saved = first.get("/api/fitting/sessions").get_json()["sessions"][0]
    assert saved["document"] == doc and saved["version"] == 1
    assert second.get("/api/fitting/sessions").get_json()["sessions"] == []
    assert (
        second.post(
            "/api/fitting/sessions/fitting-session-001",
            json={"version": 1, "document": doc},
        ).status_code
        == 409
    )
    changed = deepcopy(doc)
    changed["state"]["weights"]["HE"] = 10
    assert (
        first.post(
            "/api/fitting/sessions/fitting-session-001",
            json={"version": 0, "document": changed},
        ).status_code
        == 409
    )
    assert (
        first.get("/api/fitting/sessions").get_json()["sessions"][0]["document"] == doc
    )
    assert (
        first.post(
            "/api/fitting/sessions/fitting-session-001",
            json={"version": 1, "document": changed},
        ).status_code
        == 200
    )


def test_guest_account_session_endpoint_requires_sign_in(clients):
    guest = web.app.test_client()
    token = guest.get("/api/session").get_json()["csrf_token"]
    assert guest.get("/api/fitting/sessions").status_code == 401
    assert (
        guest.post(
            "/api/fitting/sessions/fitting-session-001",
            json={"version": 0, "document": session_document()},
            headers={"X-CSRF-Token": token},
        ).status_code
        == 401
    )
