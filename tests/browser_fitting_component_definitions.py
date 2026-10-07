"""Regression for visible fitting definition choices and submitted PFD context.

Run: python tests/browser_fitting_component_definitions.py
Uses an isolated preview and captures submissions without starting fit jobs.
"""

import json
from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview


def main():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page()
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def capture(route):
            requests.append(route.request.post_data_json)
            route.fulfill(status=400, json={"error": "Captured fitting request"})

        def state():
            return page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")

        def submit():
            count = len(requests)
            with page.expect_response(lambda response: response.url.endswith("/api/fitting")):
                page.locator("#fit-run").click()
            expect(page.locator(".toast").last).to_contain_text("Captured fitting request")
            assert len(requests) == count + 1
            return requests[-1]

        page.route("**/api/fitting", capture)
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-law option[value='constant_inverse_anchored_linear']")).to_have_text(
            "A + B/T + C h(T) + D T"
        )
        page.locator("#fit-comp1").fill("1-butanol")
        page.locator("#fit-comp1").press("Tab")
        page.locator("#fit-input").fill(json.dumps([
            {"kind": "LLE", "T_K": 300, "x1_alpha": 0.02},
        ]))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        independent = state()
        project = {
            "id": None, "filename": "ethanol-water.pfd",
            "text": "ONLINE_LOOKUP: false\nCOMPONENTS:\n    E | ethanol\n    W | water\n",
            "pfd": {
                "metadata": {}, "thermo_scopes": [],
                "components": [{"symbol": "E", "name": "ethanol"}, {"symbol": "W", "name": "water"}],
            },
        }

        # Reproduce a restored hidden PFD while component inputs describe a
        # different independent mixture. The selector must show the real basis.
        restored = {**independent, "definitionProject": project}
        page.evaluate("saved => localStorage.setItem('pfdsim.fitting.v1', JSON.stringify(saved))", restored)
        page.reload()
        expect(page.locator("#fit-project")).to_have_value("@imported-pfd")
        expect(page.locator("#fit-project option:checked")).to_have_text("ethanol-water.pfd")
        assert submit()["pfd_text"] == project["text"]

        # Independent selection clears the context and survives a reload.
        page.locator("#fit-project").select_option("")
        assert state()["definitionProject"] is None
        request = submit()
        assert request["components"] == ["1-butanol", "water"]
        assert "pfd_text" not in request and "scope" not in request
        page.reload()
        expect(page.locator("#fit-project")).to_have_value("")
        assert "pfd_text" not in submit()

        # Opening a saved session must also update the selector immediately.
        document = {
            "type": "pfdsim_fit_session", "schema_version": 1,
            "name": "Imported definition snapshot", "state": restored,
        }
        artifact = REPORT / "definition-session.json"
        artifact.write_text(json.dumps(document))
        page.locator("#fit-saved-sessions").click()
        expect(page.locator("#fit-session-file")).to_be_attached()
        page.locator("#fit-session-file").set_input_files(artifact)
        expect(page.locator("#modal")).not_to_be_visible()
        expect(page.locator("#fit-project")).to_have_value("@imported-pfd")
        assert submit()["pfd_text"] == project["text"]

        # A saved laboratory snapshot remains visible even if the local
        # laboratory library no longer contains its original ID.
        restored["definitionProject"] = {**project, "id": "removed-laboratory"}
        page.evaluate("saved => localStorage.setItem('pfdsim.fitting.v1', JSON.stringify(saved))", restored)
        page.reload()
        expect(page.locator("#fit-project")).to_have_value("removed-laboratory")
        assert submit()["pfd_text"] == project["text"]
        page.locator("#fit-project").select_option("")
        assert "pfd_text" not in submit()
        assert not errors, errors
        page.screenshot(path=REPORT / "fitting-component-definitions.png", full_page=True)
        browser.close()
    print(f"Fitting component-definition browser regression passed; report: {REPORT}")


if __name__ == "__main__":
    main()
