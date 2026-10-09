"""Opt-in browser regression for cancelling and resuming an import preview."""

import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.fitting_import_samples import WIKIPEDIA_TXY


def main():
    trace = []
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def record(response):
            if "/api/fitting/parse" in response.url:
                result = response.json()
                trace.append({"options": response.request.post_data_json.get("import_options"),
                              "ready": result.get("ready"), "issues": result.get("issues"),
                              "excluded": result.get("excluded"), "count": len(result.get("observations") or [])})

        page.on("response", record)
        try:
            page.goto(address + "/parameter-fitting")
            expect(page.locator("#fit-vapor option")).to_have_count(8)
            page.locator("#fit-input").fill("kind,T_C,x1,HE_J_mol\nHE,25,0.5,200")
            page.locator("#fit-parse").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(1)
            page.locator("#fit-input").fill(WIKIPEDIA_TXY)
            page.locator("#fit-parse").click()
            expect(page.locator("#modal")).to_be_visible()
            page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
            page.locator("#fit-parse").click()
            page.get_by_role("button", name="Use acetone as component 1").click()
            page.locator("#fit-import-pressure").fill("1")
            page.locator("#fit-import-pressure_unit").select_option("atm")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            # Selecting a unit leaves the edited pressure textbox focused.
            # Its blur/change validation must not swallow the Add click.
            expect(page.locator("#fit-import-pressure")).to_be_focused()
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(17)
            expect(page.locator("#modal")).not_to_be_visible()
            state = page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")
            assert len(state["observations"]) == 17
            assert state["observations"][1]["P_bar"] == 1.01325
            # An invalid unblurred edit must be rejected by final validation.
            page.locator("#fit-input").fill(WIKIPEDIA_TXY)
            page.locator("#fit-parse").click()
            page.locator("#fit-import-pressure").fill("1")
            page.locator("#fit-import-pressure_unit").select_option("atm")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            page.locator("#fit-import-pressure").fill("-1")
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-import-add")).to_be_disabled()
            expect(page.locator("#modal")).to_be_visible()
            expect(page.locator("#fit-table tbody tr")).to_have_count(17)
            # Correcting the input and clicking once imports the latest value.
            page.locator("#fit-import-pressure").fill("2")
            page.locator("#fit-import-pressure_unit").select_option("bar")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(33)
            state = page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")
            assert all(row["P_bar"] == 2 for row in state["observations"][17:])
            # A response for older inputs must not import after a new edit.
            page.locator("#fit-input").fill(WIKIPEDIA_TXY)
            page.locator("#fit-parse").click()
            page.locator("#fit-import-pressure").fill("1")
            page.locator("#fit-import-pressure_unit").select_option("bar")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            with page.expect_response("**/api/fitting/parse"):
                page.locator("#fit-import-pressure").press("Tab")

            def edit_during_validation(route):
                response = route.fetch()
                page.evaluate("""() => {
                    const pressure = document.getElementById('fit-import-pressure');
                    pressure.value = '3';
                    pressure.dispatchEvent(new Event('input', {bubbles:true}));
                }""")
                route.fulfill(response=response)

            page.route("**/api/fitting/parse", edit_during_validation, times=1)
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-import-pressure")).to_have_value("3")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            expect(page.locator("#modal")).to_be_visible()
            expect(page.locator("#fit-table tbody tr")).to_have_count(33)
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(49)
            state = page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")
            assert all(row["P_bar"] == 3 for row in state["observations"][33:])
            # Source-cell edits use the same validation lifecycle as conditions.
            page.locator("#fit-input").fill(WIKIPEDIA_TXY)
            page.locator("#fit-parse").click()
            page.locator("#fit-import-pressure").fill("1")
            page.locator("#fit-import-pressure_unit").select_option("bar")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            page.get_by_role("textbox", name="Source row 2 column 1", exact=True).fill("90")
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(65)
            state = page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")
            assert state["observations"][49]["T_K"] == 363.15
            assert not errors, errors
        finally:
            (REPORT / "import-trace.json").write_text(json.dumps(trace, indent=2))
            (REPORT / "import-state.json").write_text(json.dumps(page.evaluate("""() => ({
                modal: document.getElementById('modal').open,
                modalText: document.getElementById('modal-content').innerText,
                notifications: document.getElementById('notifications').innerText,
                draft: JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))
            })"""), indent=2))
            print("Import preview artifacts:", REPORT, flush=True)
            browser.close()
    print("Import preview browser checks passed.")


if __name__ == "__main__":
    main()
