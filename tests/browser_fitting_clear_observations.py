"""Real-browser coverage of selective and complete observation clearing.

Run: python tests/browser_fitting_clear_observations.py
"""

import json
from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview


def main():
    rows = [
        {
            "kind": "VLE",
            "T_K": 350,
            "P_bar": 1,
            "x1": 0.2,
            "y1": 0.4,
            "validation_only": True,
        },
        {"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.3, "y1": 0.5, "pin": True},
        {"kind": "HE", "T_K": 300, "x1": 0.2, "HE_J_mol": -20, "weight": 2},
        {"kind": "HE", "T_K": 310, "x1": 0.3, "HE_J_mol": 30, "sigma": {"HE_J_mol": 5}},
        {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2},
        {"kind": "LLE", "T_K": 300, "x1_alpha": 0.1, "x1_beta": 0.9},
        {"kind": "VLLE", "T_K": 350, "P_bar": 1},
        {"kind": "AZEOTROPE", "T_K": 350, "P_bar": 1, "x1": 0.5},
        {"kind": "UCST", "T_K": 310, "x1": 0.4},
        {"kind": "LCST", "T_K": 320, "x1": 0.6},
    ]
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

        def clear(kind):
            page.locator("#fit-clear").click()
            page.locator("#fit-clear-kind").select_option(kind)
            page.locator("#fit-clear-confirm").click()
            expect(page.locator("#modal")).not_to_be_visible()

        page.locator("#fit-input").fill(json.dumps(rows))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(10)
        original = draft()
        page.locator("#fit-clear").click()
        expect(page.locator("#modal-title")).to_have_text("Clear observations")
        expect(page.locator("#fit-clear-confirm")).to_be_disabled()
        expect(page.locator('#fit-clear-kind option[value="VLE"]')).to_contain_text(
            "(2)"
        )
        expect(page.locator('#fit-clear-kind option[value="HE"]')).to_contain_text(
            "(2)"
        )
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        assert draft() == original
        page.locator("#fit-clear").click()
        page.keyboard.press("Escape")
        assert draft() == original

        # Keep an unresolved import while clearing only selected observation types.
        unfinished = "T_K x1 HE_J_mol\n300 .2 ?"
        page.locator("#fit-input").fill(unfinished)
        page.locator("#fit-parse").click()
        expect(page.locator("#modal")).to_be_visible()
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        page.wait_for_function(
            "text => JSON.parse(localStorage.getItem('pfdsim.fitting.v1')).pendingImport?.text === text",
            arg=unfinished,
        )
        pending = draft()["pendingImport"]
        page.get_by_text("Advanced controls", exact=True).click()
        initial = {
            "12.constant": 1,
            "critical_x1.9": 0.4,
            "critical_x1.10": 0.6,
            "vlle_xa.7": 0.1,
            "vlle_gap.7": 0.8,
        }
        bounds = {
            "12.constant": [-2, 2],
            "critical_x1.9": [0.2, 0.7],
            "critical_x1.10": [0.2, 0.7],
            "vlle_xa.7": [0.01, 0.2],
        }
        page.locator("#fit-initial").fill(json.dumps(initial))
        page.locator("#fit-initial").press("Tab")
        page.locator("#fit-bounds").fill(json.dumps(bounds))
        page.locator("#fit-bounds").press("Tab")

        for kind in [
            "VLE",
            "HE",
            "GAMMA_INF",
            "LLE",
            "VLLE",
            "AZEOTROPE",
            "UCST",
            "LCST",
        ]:
            before = draft()
            expected = [row for row in before["observations"] if row["kind"] != kind]
            clear(kind)
            expect(page.locator("#fit-table tbody tr")).to_have_count(len(expected))
            after = draft()
            assert after["observations"] == expected
            assert after["pendingImport"] == pending, (
                kind,
                after["pendingImport"],
                pending,
            )
            assert after["controls"]["input"] == unfinished
            assert after["observationSets"] == before["observationSets"]
            assert after["importReports"] == before["importReports"]
            removed_ids = {
                row["id"] for row in before["observations"] if row["kind"] == kind
            }
            for field in ["initial", "bounds"]:
                old_values = json.loads(before["controls"][field])
                expected_values = {
                    key: value
                    for key, value in old_values.items()
                    if key.split(".", 1)[-1] not in removed_ids
                }
                assert json.loads(after["controls"][field]) == expected_values
            page.reload()
            expect(page.locator("#fit-data-kind option")).to_have_count(9)
            expect(page.locator("#fit-table tbody tr")).to_have_count(len(expected))
            page.locator("#fit-clear").click()
            assert page.locator(f'#fit-clear-kind option[value="{kind}"]').evaluate(
                "node => node.disabled"
            )
            page.locator("#modal-close").click()

        assert json.loads(draft()["controls"]["initial"]) == {"12.constant": 1}
        assert json.loads(draft()["controls"]["bounds"]) == {"12.constant": [-2, 2]}
        # All rows were removed selectively: historical IDs must remain reserved.
        page.locator("#fit-input").fill(json.dumps([rows[2]]))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        assert draft()["observations"][0]["id"] == "11"
        page.locator('button[aria-label="Remove observation 11"]').click()
        page.locator("#fit-input").fill(json.dumps([rows[3]]))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        assert draft()["observations"][0]["id"] == "12"

        # Malformed parameter JSON remains editable during selective clearing.
        page.get_by_text("Advanced controls", exact=True).click()
        page.locator("#fit-initial").fill('{"unfinished":')
        page.locator("#fit-initial").press("Tab")
        clear("HE")
        assert draft()["controls"]["initial"] == '{"unfinished":'
        page.locator("#fit-bounds").fill(
            json.dumps({"12.constant": [-2, 2], "critical_x1.retired": [0.1, 0.9]})
        )
        page.locator("#fit-bounds").press("Tab")
        page.locator("#fit-input").fill("unfinished paste")
        clear("all")
        state = draft()
        assert (
            state["observations"]
            == state["observationSets"]
            == state["importReports"]
            == []
        )
        assert state["controls"]["input"] == "" and state["pendingImport"] is None
        assert json.loads(state["controls"]["bounds"]) == {"12.constant": [-2, 2]}
        # All-clear also resets current manual input and permits fresh IDs.
        page.locator("#fit-input-method").select_option("manual")
        page.locator("#fit-data-kind").select_option("GAMMA_INF")
        page.locator("#fit-manual-gamma1_inf").fill("2")
        page.locator("#fit-manual-gamma2_inf").fill("3")
        page.locator("#fit-data-kind").select_option("HE")
        page.locator("#fit-manual-temperature").fill("300")
        page.locator("#fit-manual-x1").fill(".2")
        page.locator("#fit-manual-enthalpy").fill("20")
        clear("all")
        expect(page.locator("#fit-manual-temperature")).to_have_value("")
        expect(page.locator("#fit-manual-x1")).to_have_value("")
        expect(page.locator("#fit-manual-enthalpy")).to_have_value("")
        page.locator("#fit-data-kind").select_option("GAMMA_INF")
        expect(page.locator("#fit-manual-gamma1_inf")).to_have_value("")
        expect(page.locator("#fit-manual-gamma2_inf")).to_have_value("")
        page.locator("#fit-data-kind").select_option("HE")
        page.locator("#fit-input-method").select_option("paste")
        page.locator("#fit-input").fill(json.dumps([rows[2]]))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        assert draft()["observations"][0]["id"] == "1"
        assert not errors, errors
        page.screenshot(path=str(REPORT / "clear-observations.png"), full_page=True)
        browser.close()
    print(f"Clear-observations browser regression passed; report: {REPORT}")


if __name__ == "__main__":
    main()
