"""Opt-in browser coverage for wide-table interpretation and saved fit sessions."""

from pathlib import Path
import json
import shutil
import sys

from playwright.sync_api import sync_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.test_repeated_series_import import HE_PASTE


def saved_state(page):
    return page.evaluate("JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))")


def open_sessions(page):
    page.locator("#fit-saved-sessions").click()
    expect(page.locator("#fit-session-name")).to_be_visible()


def open_saved(page, name):
    open_sessions(page)
    option = page.locator("#fit-session-list option").filter(has_text=name).first
    page.locator("#fit-session-list").select_option(option.get_attribute("value"))
    page.locator("#fit-session-open").click()
    expect(page.locator("#modal")).not_to_be_visible()


def main():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        context = browser.new_context(viewport={"width": 1500, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-vapor option")).to_have_count(8)
        page.locator("#fit-data-kind").select_option("HE")
        page.locator("#fit-input").fill(HE_PASTE)
        page.locator("#fit-parse").click()
        expect(page.locator("#modal")).to_be_visible()
        expect(page.locator("#fit-import-add")).to_be_disabled()
        assert page.locator("#fit-table tbody tr").count() == 0
        page.locator("#fit-import-repeated").click()
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        page.locator("#fit-import-temperature_unit").select_option("K")
        page.locator("#fit-import-enthalpy_unit").select_option("J/mol")
        page.locator("#fit-import-composition_basis").select_option("mole_fraction")
        for index in range(5):
            page.locator(f"#fit-import-series-{index}-temperature").fill(
                str(298.15 + 10 * index)
            )
        page.locator("#fit-import-series-4-validation_only").check()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(95, timeout=15000)
        state = saved_state(page)
        assert len(state["observationSets"]) == 5
        assert len({row["group"] for row in state["observations"]}) == 5
        assert sum(row["validation_only"] for row in state["observations"]) == 19
        assert state["observations"][0]["HE_J_mol"] == 290.4
        page.locator("#fit-model").select_option("UNIQUAC")
        page.locator("#fit-r1").fill("2.5")
        page.locator("#fit-q1").fill("2.2")
        page.locator("#fit-r2").fill("3.1")
        page.locator("#fit-q2").fill("2.8")
        open_sessions(page)
        page.locator("#fit-session-name").fill("Five HE temperatures")
        page.locator("#fit-session-save").click()
        expect(page.locator("#modal")).to_contain_text("Saved Five HE temperatures")
        with page.expect_download() as received:
            page.locator("#fit-session-download").click()
        artifact = REPORT / "complete-session.json"
        received.value.save_as(artifact)
        document = json.loads(artifact.read_text())
        assert document["state"]["controls"]["model"] == "UNIQUAC"
        assert len(document["state"]["observations"]) == 95
        page.locator("#modal-close").click()
        page.locator("#fit-clear").click()
        page.locator("#fit-reset").click()
        open_saved(page, "Five HE temperatures")
        expect(page.locator("#fit-table tbody tr")).to_have_count(95)
        expect(page.locator("#fit-model")).to_have_value("UNIQUAC")
        expect(page.locator("#fit-r1")).to_have_value("2.5")
        page.reload()
        expect(page.locator("#fit-table tbody tr")).to_have_count(95)
        page.locator("#fit-clear").click()
        open_sessions(page)
        page.locator("#fit-session-file").set_input_files(artifact)
        expect(page.locator("#fit-table tbody tr")).to_have_count(95)
        # An unresolved repeated-series popup can be resumed after reload and
        # is included when saving the complete fitting state.
        page.locator("#fit-clear").click()
        page.locator("#fit-input").fill(HE_PASTE)
        page.locator("#fit-parse").click()
        page.locator("#fit-import-repeated").click()
        page.locator("#fit-import-series-0-temperature").fill("300")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        assert (
            saved_state(page)["pendingImport"]["options"]["series"][0]["temperature"]
            == 300
        )
        page.reload()
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-series-0-temperature")).to_have_value("300")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        # Account saves are recovered in a fresh browser context with the same
        # authenticated session, without relying on the first context's storage.
        page.locator("#account-button").click()
        page.locator("#modal").get_by_role(
            "button", name="Create account", exact=True
        ).click()
        page.locator('#modal input[name="username"]').fill("fit-session-browser")
        page.locator('#modal input[name="password"]').fill(
            "Long saved-fit browser password"
        )
        page.locator('#modal form button[type="submit"]').click()
        expect(page.locator("#account-button")).to_have_text(
            "fit-session-browser", timeout=15000
        )
        open_sessions(page)
        page.locator("#fit-session-name").fill("Account saved wide table")
        page.locator("#fit-session-copy").click()
        expect(page.locator("#modal")).to_contain_text("to your account and browser")
        page.locator("#modal-close").click()
        second = browser.new_context(viewport={"width": 1500, "height": 1000})
        second.add_cookies(context.cookies())
        remote = second.new_page()
        remote.goto(address + "/parameter-fitting")
        expect(remote.locator("#account-button")).to_have_text("fit-session-browser")
        open_saved(remote, "Account saved wide table")
        expect(remote.locator("#fit-input")).to_have_value(HE_PASTE)
        remote.locator("#fit-parse").click()
        expect(remote.locator("#fit-import-series-0-temperature")).to_have_value("300")
        remote.locator("#modal").get_by_role(
            "button", name="Cancel", exact=True
        ).click()
        page.screenshot(path=REPORT / "saved-fit-library.png", full_page=True)
        assert not errors, errors
        browser.close()
    print("Repeated-series and saved-session browser checks passed. Artifacts:", REPORT)


if __name__ == "__main__":
    main()
