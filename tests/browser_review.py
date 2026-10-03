"""Focused browser regressions for draft safety and truthful calculation state.

Run with Playwright and Chrome/Chromium: python tests/browser_review.py
The preview is private; calculation responses are mocked to avoid costly runs.
"""

import os
import shutil
import tempfile

from browser_web import REPORT, label_input, preview
from playwright.sync_api import expect, sync_playwright

SOURCE = """PROCESS: Review laboratory
THERMO_METHOD: NRTL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    local | method=NRTL-BV
COMPONENTS:
    A | ethanol
    B | water
UNIT U : Pipe
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
    length = 1 [m]
    diameter = 0.1 [m]
UNIT V : Pump
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
    P_out = 2 [bar]
"""


def run():
    os.environ["PFDSIM_WEB_DATA"] = tempfile.mkdtemp(prefix="pfdsim-review-web-")
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.goto(address + "/editor")
        expect(page.locator(".palette-item")).to_have_count(33)
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill(SOURCE)
        page.click("#apply-source")
        expect(page.locator("#source-status")).to_have_text("Source synchronized")
        page.locator("[data-view=diagram]").click()
        page.click("#undo-button")
        expect(page.locator("[data-unit]")).to_have_count(0)
        page.click("#redo-button")
        expect(page.locator("[data-unit]")).to_have_count(2)
        expect(page.locator("#source-status")).to_have_text("Source synchronized")

        # Geometry labels must not offer units that the solver never converts.
        page.locator("[data-unit=U]").press("Enter")
        geometry_units = (
            page.locator("[data-setting=length] select")
            .locator("option")
            .all_text_contents()
        )
        assert not {"mm", "cm", "ft"}.intersection(geometry_units), geometry_units

        # Dropping equipment and connecting ports must protect inspector drafts.
        label_input(page.locator("#inspector-content"), "Id").fill("DraftName")
        page.locator("#flowsheet").evaluate("""node => {
            const transfer = new DataTransfer();
            transfer.setData('pfdsim/unit', 'Pump');
            node.dispatchEvent(new DragEvent('drop', {bubbles:true, dataTransfer:transfer}));
        }""")
        expect(page.locator("#modal-title")).to_have_text("Unapplied inspector changes")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        expect(label_input(page.locator("#inspector-content"), "Id")).to_have_value(
            "DraftName"
        )
        expect(page.locator("[data-unit]")).to_have_count(2)
        page.locator("[data-unit=U] [data-port=out]").press("Enter")
        expect(page.locator("#modal-title")).to_have_text("Unapplied inspector changes")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill(SOURCE + "\n# source draft\n")
        page.locator("#code-editor").press("Control+Enter")
        expect(page.locator("#modal-title")).to_have_text("Unapplied inspector changes")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()

        # A terminal 404 must release controls; transient errors retain the job.
        page.evaluate("""() => {
            const key = 'pfdsim.laboratories.v1';
            const library = JSON.parse(localStorage.getItem(key));
            const id = JSON.parse(localStorage.getItem('pfdsim.last.v1'));
            library[id].job = {id:'expired-review-run', text:library[id].text};
            localStorage.setItem(key, JSON.stringify(library));
        }""")
        page.reload()
        expect(page.locator("#run-status")).to_contain_text("unavailable or expired")
        expect(page.locator("#run-button")).to_be_enabled()
        expect(page.locator("#cancel-run")).to_be_hidden()

        # Stored results must gain a visible warning when the source changes.
        page.route(
            "**/api/simulate",
            lambda route: route.fulfill(json={"job_id": "review-result"}, status=202),
        )
        output = {
            "converged": True,
            "iterations": 1,
            "mass_balance_error": None,
            "energy_balance_error": None,
            "results": {},
            "pfr_content": "review results",
        }
        page.route(
            "**/api/jobs/review-result",
            lambda route: route.fulfill(
                json={
                    "job": {"status": "completed", "progress": [], "output": output},
                }
            ),
        )
        page.click("#run-button")
        expect(page.locator("#run-status")).to_contain_text("complete · converged")
        for name in ("Mass balance error", "Energy balance error"):
            expect(page.locator(".metric").filter(has_text=name).locator("strong")).to_have_text("—")
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill("unfinished invalid source draft")
        expect(page.locator("#result-indicator")).to_have_text("•")
        page.locator("[data-view=results]").click()
        expect(page.locator("#result-content")).to_contain_text(
            "earlier flowsheet revision"
        )
        page.screenshot(path=REPORT / "review-results.png", full_page=True)

        page.goto(address + "/vle-chart")
        expect(page.locator("#chart-method option").first).to_be_attached()
        page.locator("#atlas-pfd-file").set_input_files(
            {
                "name": "review.pfd",
                "mimeType": "text/plain",
                "buffer": SOURCE.encode(),
            }
        )
        expect(page.locator("#chart-source")).to_have_value("imported")
        expect(page.locator("#chart-online")).not_to_be_checked()
        expect(page.locator("#chart-online")).to_be_disabled()
        page.locator("#chart-source").select_option("independent")
        expect(page.locator("#chart-online")).to_be_checked()
        page.locator("#chart-source").select_option("imported")
        expect(page.locator("#scope-field")).to_be_visible()
        expect(page.locator("#chart-online")).not_to_be_checked()
        page.get_by_text("Advanced options", exact=True).click()
        page.locator("#chart-scope").select_option("local")
        expect(page.locator("#chart-model-options")).to_contain_text(
            "Named scopes use model defaults"
        )
        expect(page.locator("#chart-model-options input")).to_have_count(0)
        page.locator("#chart-scope").select_option("global")
        page.locator("#chart-method").select_option("NRTL-BV")
        expect(
            page.locator("#chart-model-options [data-setting=correlation]")
        ).to_be_visible()

        # Saved laboratories use the applied model even with invalid source drafts.
        laboratory = page.evaluate("JSON.parse(localStorage.getItem('pfdsim.last.v1'))")
        page.locator("#chart-source").select_option(laboratory)
        expect(page.locator("#definition-description")).to_contain_text(
            "Unapplied source edits are excluded"
        )
        submitted = []

        def submit(route):
            submitted.append(route.request.post_data_json)
            route.fulfill(json={"job_id": "expired-review-chart"}, status=202)

        page.route("**/api/vle-chart", submit)
        page.route(
            "**/api/jobs/expired-review-chart",
            lambda route: route.fulfill(
                json={"error": "Run not found or expired."},
                status=404,
            ),
        )
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_contain_text("unavailable or expired")
        assert "unfinished invalid source draft" not in submitted[0]["text"]
        assert "UNIT U\n    TYPE: Pipe" in submitted[0]["text"]
        expect(page.locator("#generate-chart")).to_be_enabled()
        page.screenshot(path=REPORT / "review-atlas.png", full_page=True)

        # A failed account-list request cannot prevent access to browser drafts.
        session = page.request.get(address + "/api/session").json()
        response = page.request.post(
            address + "/api/account/register",
            data={
                "username": "ReviewBrowser",
                "password": "review browser password",
            },
            headers={"X-CSRF-Token": session["csrf_token"]},
        )
        assert response.status == 200, response.text()
        page.route(
            "**/api/flowsheets",
            lambda route: route.fulfill(
                json={"error": "temporary account storage failure"},
                status=503,
            ),
        )
        held_saves = []
        page.route("**/api/flowsheets/*", lambda route: held_saves.append(route))
        page.goto(address + "/editor")
        expect(page.locator(".palette-item")).to_have_count(33)
        expect(page.locator("#project-name")).to_have_text("Review laboratory")
        page.locator("[data-view=code]").click()
        expect(page.locator("#code-editor")).to_have_value(
            "unfinished invalid source draft"
        )
        expect(page.locator("#cloud-save-status")).to_have_text(
            "Saving to your account…"
        )
        assert held_saves
        page.locator("#code-editor").fill("newer unsynced source draft")
        held_saves[0].fulfill(json={"version": 1, "updated": 1})
        page.wait_for_function("""() => {
            const id = JSON.parse(localStorage.getItem('pfdsim.last.v1'));
            return JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id].cloudVersion === 1;
        }""")
        expect(page.locator("#cloud-save-status")).to_have_text("Account sync pending…")
        page.locator("#code-editor").fill("UNIT Empty : Pump\n")
        page.click("#apply-source")
        expect(page.locator("#source-status")).to_have_text("Source synchronized")
        page.locator("[data-view=diagram]").click()
        page.locator("[data-unit=Empty]").press("Enter")
        page.get_by_role("button", name="Add standard ports", exact=True).click()
        expect(page.locator("[data-unit=Empty] [data-port]")).to_have_count(2)
        assert not failures, failures
        page.unroute_all(behavior="ignoreErrors")
        browser.close()
        print(
            "PASS: source undo/redo, geometry units, inspector draft guards, expired job recovery, unavailable balance metrics, stale-result warnings, imported definitions/policies, model options, applied atlas inputs, account-sync failure recovery, truthful concurrent autosave status, and adding standard ports"
        )
        print("Browser artifacts:", REPORT)


if __name__ == "__main__":
    print("Review browser report:", REPORT, flush=True)
    run()
