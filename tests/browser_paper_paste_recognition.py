"""Opt-in browser checks for inferred paper columns and per-column units."""

from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.fitting_import_samples import OCR_ACETONE_WATER_HE


def observations(page):
    return page.evaluate(
        "JSON.parse(localStorage.getItem('pfdsim.fitting.v1')).observations"
    )


def main():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-vapor option")).to_have_count(8)

        page.locator("#fit-input").fill(
            "VLE at 100 kPa\nT_K\tx₁ / mole %\ty₂ / mole fraction\n330\t20\t0.4\n340\t50\t0.2"
        )
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(2)
        first = observations(page)[0]
        assert first["x1"] == 0.2 and first["y1"] == 0.6

        page.locator("#fit-clear").click()
        page.locator("#fit-input").fill(
            "??\tx?\tv?\t???\n350\t0.1\t0.3\t80\n355\t0.4\t0.65\t85\n360\t0.8\t0.9\t90"
        )
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-column-0")).to_have_value("temperature")
        expect(page.locator("#fit-import-column-3")).to_have_value("pressure")
        expect(page.locator("#fit-import-add")).to_be_disabled()
        page.locator("#fit-import-pressure_unit").select_option("kpa")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(3)
        assert observations(page)[0]["P_bar"] == 0.8

        page.locator("#fit-clear").click()
        page.locator("#fit-input").fill(
            "x1\tH^E / J mol⁻¹\tH^E / kJ mol⁻¹\n0.2\t200\t0.25\n0.5\t400\t0.45"
        )
        page.locator("#fit-parse").click()
        page.locator("#fit-import-repeated").click()
        expect(page.locator("#fit-import-series-count")).to_have_value("2")
        page.locator("#fit-import-temperature_unit").select_option("K")
        page.locator("#fit-import-series-0-temperature").fill("300")
        page.locator("#fit-import-series-1-temperature").fill("320")
        page.locator("#fit-import-series-1-temperature").press("Tab")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(4)
        assert [row["HE_J_mol"] for row in observations(page)] == [200, 400, 250, 450]

        page.locator("#fit-clear").click()
        page.locator("#fit-comp1").fill("acetone")
        page.locator("#fit-input").fill(OCR_ACETONE_WATER_HE)
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-kind")).to_have_value("HE")
        expect(page.locator("#fit-import-enthalpy_unit")).to_have_value("J/mol")
        expect(page.locator("#fit-import-temperature_unit")).to_have_value("K")
        expect(page.locator("#modal")).to_contain_text("Unassigned HE values: -40.5")
        expect(page.locator("#fit-import-add")).to_be_disabled()
        with page.expect_response(
            lambda response: response.url.endswith("/api/fitting/parse")
        ) as refreshed:
            page.locator("#fit-import-repeated").click()
        assert refreshed.value.json()["success"]
        expect(page.locator("#fit-import-series-count")).to_have_value("5")
        for index, temperature in enumerate([283.15, 298.15, 323.15, 343.15, 363.15]):
            expect(
                page.locator(f"#fit-import-series-{index}-temperature")
            ).to_have_value(str(temperature))
            expect(
                page.locator(f"#fit-import-series-{index}-temperature_unit")
            ).to_have_value("K")
        expect(page.locator("#fit-import-add")).to_be_disabled()
        page.screenshot(path=REPORT / "ragged-he-preview-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#fit-import-kind").scroll_into_view_if_needed()
        page.screenshot(path=REPORT / "ragged-he-preview-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1500, "height": 1000})
        for index in [0, 1, 3, 9, 10]:
            with page.expect_response(
                lambda response: response.url.endswith("/api/fitting/parse")
            ) as refreshed:
                page.locator(f'#modal input[data-row="{index}"]').uncheck()
            assert refreshed.value.json()["success"]
            expect(
                page.locator(f'#modal input[data-row="{index}"]')
            ).not_to_be_checked()
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(90)
        assert sorted({row["T_K"] for row in observations(page)}) == [
            283.15,
            298.15,
            323.15,
            343.15,
            363.15,
        ]

        page.screenshot(path=REPORT / "paper-paste-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=REPORT / "paper-paste-mobile.png", full_page=True)
        assert not errors, errors
        browser.close()
    print("Paper-paste browser checks passed. Artifacts:", REPORT)


if __name__ == "__main__":
    main()
