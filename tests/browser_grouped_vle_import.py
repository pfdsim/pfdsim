"""Grouped Txy/Pxy tables, ignored modeled values and stale HE import state.

Run: python tests/browser_grouped_vle_import.py
"""

from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.fitting_import_samples import PRESSURE_GROUPED_TXY
from tests.test_grouped_vle_import import COUNTS, PRESSURES, grouped_table


def main():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
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
            expect(page.locator("#modal")).to_be_visible()

        def cancel():
            page.locator("#modal").get_by_role(
                "button", name="Cancel", exact=True
            ).click()
            page.wait_for_function(
                "JSON.parse(localStorage.getItem('pfdsim.fitting.v1')).pendingImport !== null"
            )

        def clear():
            page.locator("#fit-clear").click()
            page.locator("#fit-clear-kind").select_option("all")
            page.locator("#fit-clear-confirm").click()

        cases = [
            (PRESSURE_GROUPED_TXY, "T", False, "AUTO"),
            (grouped_table("T", junk=True, multiline=True), "T", True, "VLE"),
            (grouped_table("P", junk=True), "P", True, "AUTO"),
            (grouped_table("P", format="html", junk=True), "P", True, "VLE"),
        ]
        for text, axis, junk, kind in cases:
            clear()
            page.locator("#fit-data-kind").select_option(kind)
            paste(text)
            expect(page.locator("#fit-import-kind")).to_have_value("VLE")
            expect(page.locator("#fit-import-series-count")).to_have_value("5")
            expect(page.locator("#fit-import-series-layout")).to_have_value("detected")
            expect(page.locator("#fit-import-add")).to_be_enabled()
            for group in range(5):
                condition = "pressure" if axis == "T" else "temperature"
                expect(
                    page.locator(f"#fit-import-series-{group}-{condition}")
                ).to_have_value(
                    str(PRESSURES[group]) if axis == "T" else str(300 + 10 * group)
                )
                expect(
                    page.locator(f"#fit-import-series-{group}-{condition}_unit")
                ).to_have_value("bar" if axis == "T" else "K")
                if junk:
                    index = 4 * group + 1
                    expect(page.locator(f"#fit-import-column-{index}")).to_have_value(
                        "ignore"
                    )
                    expect(
                        page.locator(f"#fit-import-column-series-{index}")
                    ).to_have_value("ignore")
            if junk and axis == "P":
                # Switching out of detected mode must not silently keep auto grouping.
                page.locator("#fit-import-repeated").click()
                expect(page.locator("#fit-import-add")).to_be_disabled()
                page.locator("#fit-import-repeated").click()
                expect(page.locator("#fit-import-add")).to_be_enabled()
                page.locator("#fit-import-series-layout").select_option("PxyTriples")
                expect(page.locator("#fit-import-series-count")).to_have_value("5")
                expect(page.locator("#fit-import-column-1")).to_have_value("ignore")
                expect(page.locator("#fit-import-add")).to_be_enabled()
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(75)
            state = draft()
            assert all(
                row["kind"] == "VLE" and "HE_J_mol" not in row
                for row in state["observations"]
            )
            assert [len(group["ids"]) for group in state["observationSets"]] == COUNTS
            report = state["importReports"][-1]
            assert report["original_text"] == text
            assert [
                series["observations"] for series in report["series_reports"]
            ] == COUNTS
            if axis == "T":
                assert sorted(
                    {row["P_bar"] for row in state["observations"]}
                ) == sorted(PRESSURES)
            else:
                assert sorted({row["T_K"] for row in state["observations"]}) == [
                    300,
                    310,
                    320,
                    330,
                    340,
                ]
                for group in range(5):
                    index = report["series_reports"][group]["observation_indices"][0]
                    expected = [329.85, 323.75, 315.25, 305.55, 289.55][group] / 100
                    assert abs(state["observations"][index]["P_bar"] - expected) < 1e-12

        # Unitless group captions remain blocked until their units are chosen.
        for axis in ["T", "P"]:
            clear()
            page.locator("#fit-data-kind").select_option("AUTO")
            symbol = "P" if axis == "T" else "T"
            values = (
                PRESSURES if axis == "T" else [300 + 10 * group for group in range(5)]
            )
            paste(
                grouped_table(axis, captions=[f"{symbol}={value}" for value in values])
            )
            expect(page.locator("#fit-import-series-count")).to_have_value("5")
            expect(page.locator("#fit-import-add")).to_be_disabled()
            key = "pressure_unit" if axis == "T" else "temperature_unit"
            page.locator(f"#fit-import-{key}").select_option(
                "bar" if axis == "T" else "K"
            )
            expect(page.locator("#fit-import-add")).to_be_enabled()
            page.locator("#fit-import-add").click()
            expect(page.locator("#fit-table tbody tr")).to_have_count(75)

        # Reproduce an old HE layout saved against this VLE paste, then recover it
        # through each data-type control without losing a source-cell correction.
        clear()
        page.locator("#fit-data-kind").select_option("AUTO")
        paste(PRESSURE_GROUPED_TXY)
        page.get_by_role("textbox", name="Source row 1 column 1", exact=True).fill(
            "330.85"
        )
        page.get_by_role("textbox", name="Source row 1 column 1", exact=True).press(
            "Tab"
        )
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-series-layout").select_option("HE")
        expect(page.locator("#fit-import-kind")).to_have_value("HE")
        expect(page.locator("#fit-import-add")).to_be_disabled()
        cancel()
        assert draft()["pendingImport"]["options"]["kind"] == "HE"
        # Re-pasting identical text is fresh input, rather than resuming the HE draft.
        page.locator("#fit-input").fill(PRESSURE_GROUPED_TXY)
        page.locator("#fit-input").dispatch_event("input")
        assert draft()["pendingImport"] is None
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-kind")).to_have_value("VLE")
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.get_by_role("textbox", name="Source row 1 column 1", exact=True).fill("330.85")
        page.get_by_role("textbox", name="Source row 1 column 1", exact=True).press("Tab")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-series-layout").select_option("HE")
        cancel()
        page.locator("#fit-data-kind").select_option("VLE")
        retained = draft()["pendingImport"]["options"]
        assert "mapping" not in retained and "series" not in retained
        assert retained["cell_edits"][0]["value"] == "330.85"
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-kind")).to_have_value("VLE")
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-series-layout").select_option("HE")
        expect(page.locator("#fit-import-kind")).to_have_value("HE")
        held = []

        def hold_reinterpretation(route):
            held.append((route, route.fetch()))

        page.route("**/api/fitting/parse", hold_reinterpretation)
        page.locator("#fit-import-kind").select_option("VLE")
        cancel()
        pending = draft()["pendingImport"]["options"]
        assert (
            pending["kind"] == "VLE"
            and "mapping" not in pending
            and "series" not in pending
        )
        assert held
        for route, response in held:
            route.fulfill(response=response)
        page.unroute("**/api/fitting/parse", hold_reinterpretation)
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        expect(
            page.get_by_role("textbox", name="Source row 1 column 1", exact=True)
        ).to_have_value("330.85")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        # Auto detect must also stop inheriting a previously selected HE kind.
        page.locator("#fit-import-series-layout").select_option("HE")
        cancel()
        page.locator("#fit-data-kind").select_option("AUTO")
        assert "kind" not in draft()["pendingImport"]["options"]
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-kind")).to_have_value("VLE")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        cancel()
        page.reload()
        expect(page.locator("#fit-data-summary")).to_contain_text("0 observations")
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-kind")).to_have_value("VLE")
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(75)
        assert draft()["observations"][0]["T_K"] == 330.85
        assert not errors, errors
        page.screenshot(path=str(REPORT / "grouped-vle.png"), full_page=True)
        browser.close()
    print(f"Grouped VLE browser checks passed; report: {REPORT}")


if __name__ == "__main__":
    main()
