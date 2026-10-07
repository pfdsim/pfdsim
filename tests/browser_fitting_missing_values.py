"""Focused real-browser regression for missing values and PDF table editing.

Run: python tests/browser_fitting_missing_values.py
Uses the isolated preview lifecycle; never publishes or changes runtime tables.
"""

import json
from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.test_fitting_missing_values import MARKERS


def main():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-data-kind option")).to_have_count(9)

        def draft():
            return page.evaluate(
                "JSON.parse(localStorage.getItem('pfdsim.fitting.v1'))"
            )

        def paste(text):
            page.locator("#fit-input").fill(text)
            page.locator("#fit-parse").click()

        paste(
            json.dumps(
                [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2, "gamma2_inf": 3}]
            )
        )
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        gamma2 = page.get_by_role(
            "textbox", name="Observation 1 gamma2_inf", exact=True
        )
        for marker in MARKERS:
            gamma2.fill(marker)
            assert gamma2.evaluate("node => node.checkValidity()")
            assert "gamma2_inf" not in draft()["observations"][0], marker
        gamma2.fill("-1.2e-3")
        assert draft()["observations"][0]["gamma2_inf"] == -0.0012
        gamma2.fill("bad number")
        assert not gamma2.evaluate("node => node.checkValidity()")
        page.reload()
        gamma2 = page.get_by_role(
            "textbox", name="Observation 1 gamma2_inf", exact=True
        )
        expect(gamma2).to_have_value("bad number")
        assert not gamma2.evaluate("node => node.checkValidity()")
        gamma2.fill("None")
        page.reload()
        expect(
            page.get_by_role("textbox", name="Observation 1 gamma2_inf", exact=True)
        ).to_have_value("")
        assert "gamma2_inf" not in draft()["observations"][0]

        # Submit without blurring: the request must contain the latest edit.
        requests = []

        def capture_request(route):
            requests.append(route.request.post_data_json)
            route.fulfill(status=400, json={"error": "Captured regression request"})

        page.route("**/api/fitting", capture_request)
        page.get_by_role("textbox", name="Observation 1 gamma1_inf", exact=True).fill(
            "4e0"
        )
        page.locator("#fit-run").click()
        expect(page.locator(".toast")).to_contain_text("Captured regression request")
        assert requests[-1]["observations"][0]["gamma1_inf"] == 4
        assert "gamma2_inf" not in requests[-1]["observations"][0]

        # Manual entry uses the same missing-value semantics.
        page.locator("#fit-input-method").select_option("manual")
        page.locator("#fit-data-kind").select_option("GAMMA_INF")
        page.locator("#fit-manual-temperature").fill("300")
        page.locator("#fit-manual-temperature_unit").select_option("K")
        page.locator("#fit-manual-gamma1_inf").fill("2")
        page.locator("#fit-manual-gamma2_inf").fill("not measured")
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(2)
        assert "gamma2_inf" not in draft()["observations"][1]

        # Required missing cells can be corrected in the preview and survive cancellation.
        page.locator("#fit-input-method").select_option("paste")
        page.locator("#fit-data-kind").select_option("AUTO")
        paste("kind,T_K,x1,HE_J_mol\nHE,300,.2,None")
        expect(page.locator("#modal")).to_be_visible()
        cell = page.get_by_role("textbox", name="Source row 1 column 4", exact=True)
        expect(cell).to_have_value("None")
        cell.fill("-20")
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        assert draft()["pendingImport"]["options"]["cell_edits"][-1]["value"] == "-20"
        page.reload()
        expect(page.locator("#fit-table tbody tr")).to_have_count(2)
        page.locator("#fit-parse").click()
        # Canonical imports with a completed correction can be added directly.
        expect(page.locator("#fit-table tbody tr")).to_have_count(3)
        assert draft()["observations"][2]["HE_J_mol"] == -20
        assert "None" in draft()["importReports"][-1]["original_text"]

        # Ambiguous PDF signs remain blocked until the edited row is confirmed.
        paste("T_K x1 HE_J_mol weight pin_tolerance\n300 .2 - 20 - .00001")
        expect(page.locator("#fit-import-add")).to_be_disabled()
        expect(page.locator("#modal")).to_contain_text("Unassigned values")
        for column, value in enumerate(["300", ".2", "-20", "?", ".00001"], 1):
            page.get_by_role(
                "textbox", name=f"Source row 1 column {column}", exact=True
            ).fill(value)
        page.get_by_role(
            "button", name="Confirm alignment of source row 1", exact=True
        ).click()
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(4)
        assert draft()["observations"][3]["HE_J_mol"] == -20
        assert draft()["observations"][3]["weight"] == 1

        # A flattened line asks for both dimensions, then supports both orders.
        page.locator("#fit-data-kind").select_option("GAMMA_INF")
        paste("300 2 ? 310 None 3")
        expect(page.locator("#modal")).to_contain_text("row and column counts")
        expect(page.locator("#fit-import-column_count")).to_have_value("")
        expect(page.locator("#fit-import-row_count")).to_have_value("")
        page.locator("#fit-import-row_count").fill("2")
        page.locator("#fit-import-row_count").press("Tab")
        expect(page.locator("#fit-import-row_count")).to_have_value("2")
        page.locator("#fit-import-column_count").fill("3")
        page.locator("#fit-import-column_count").press("Tab")
        expect(page.locator("#fit-import-column-2")).to_be_visible()
        page.locator("#fit-import-temperature_unit").select_option("K")
        for column, role in enumerate(["temperature", "gamma1_inf", "gamma2_inf"]):
            page.locator(f"#fit-import-column-{column}").select_option(role)
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(6)
        assert "gamma2_inf" not in draft()["observations"][4]
        assert "gamma1_inf" not in draft()["observations"][5]

        paste("320 330 2 3 4 5")
        page.locator("#fit-import-row_count").fill("2")
        page.locator("#fit-import-row_count").press("Tab")
        page.locator("#fit-import-column_count").fill("3")
        page.locator("#fit-import-column_count").press("Tab")
        expect(page.locator("#fit-import-column-2")).to_be_visible()
        page.locator("#fit-import-layout").select_option("columns")
        expect(
            page.get_by_role("textbox", name="Source row 2 column 1", exact=True)
        ).to_have_value("330")
        page.locator("#fit-import-temperature_unit").select_option("K")
        for column, role in enumerate(["temperature", "gamma1_inf", "gamma2_inf"]):
            page.locator(f"#fit-import-column-{column}").select_option(role)
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(8)
        assert [row["T_K"] for row in draft()["observations"][-2:]] == [320, 330]

        # LLE manual entry accepts either measured branch, with no fabricated
        # value for the missing branch, and still requires one endpoint.
        page.locator("#fit-input-method").select_option("manual")
        page.locator("#fit-data-kind").select_option("LLE")
        page.locator("#fit-manual-temperature").fill("300")
        page.locator("#fit-manual-temperature_unit").select_option("K")
        page.locator("#fit-manual-x1_alpha").fill(".02")
        page.locator("#fit-manual-x1_beta").fill("—")
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(9)
        assert draft()["observations"][-1]["x1_alpha"] == 0.02
        assert "x1_beta" not in draft()["observations"][-1]
        page.locator("#fit-manual-temperature").fill("300")
        page.locator("#fit-manual-x1_alpha").fill("None")
        page.locator("#fit-manual-x1_beta").fill(".8")
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(10)
        assert draft()["observations"][-1]["x1_beta"] == 0.8
        assert "x1_alpha" not in draft()["observations"][-1]
        page.locator("#fit-manual-temperature").fill("300")
        page.locator("#fit-manual-x1_beta").fill("?")
        page.locator("#fit-parse").click()
        expect(page.locator(".toast")).to_contain_text("x1_alpha or x1_beta")
        expect(page.locator("#fit-table tbody tr")).to_have_count(10)

        page.locator("#fit-input-method").select_option("paste")
        page.locator("#fit-data-kind").select_option("AUTO")
        paste("kind,T_K,x1_alpha,x1_beta\nLLE,310,.03,None\nLLE,320,?,.75")
        expect(page.locator("#fit-table tbody tr")).to_have_count(12)
        assert "x1_beta" not in draft()["observations"][-2]
        assert "x1_alpha" not in draft()["observations"][-1]
        assert not errors, errors
        page.screenshot(path=str(REPORT / "fitting-missing-values.png"), full_page=True)
        browser.close()
    print(f"Missing-value browser regression passed; report: {REPORT}")


if __name__ == "__main__":
    main()
