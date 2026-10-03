"""Real constant-pressure butanol/water VLLE, including standalone exports."""

import json
import os
import shutil
import tempfile

from browser_web import REPORT, preview
from playwright.sync_api import expect, sync_playwright


def run():
    os.environ["PFDSIM_WEB_DATA"] = tempfile.mkdtemp(prefix="pfdsim-binary-vlle-web-")
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.goto(address + "/vle-chart")
        expect(page.locator("#chart-method")).to_have_value("UNIFAC")
        page.locator("#chart-method").select_option("NRTL")
        page.locator("#chart-type").select_option("VLLE")
        page.locator("#comp1").fill("1-butanol")
        page.locator("#comp2").fill("water")
        expect(page.locator("#pressure-field")).to_be_visible()
        expect(page.locator("#temperature-field")).to_be_hidden()
        page.get_by_text("Advanced options", exact=True).click()
        expect(page.locator("#binodal-minimum-field")).to_be_visible()
        page.locator("#chart-online").uncheck()
        page.locator("#chart-points").fill("10")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text(
            "Diagram complete.", timeout=120000
        )
        expect(page.locator("#chart-frame .chart-line")).to_have_count(4)
        expect(page.locator("#chart-information")).to_contain_text(
            "Constant pressure; temperature varies"
        )
        with page.expect_download() as exported:
            page.click("#export-chart")
        data_path = REPORT / "butanol-water-vlle.json"
        exported.value.save_as(data_path)
        chart = json.loads(data_path.read_text())
        assert chart["pressure_bar"] == 1
        assert 92 < chart["heteroazeotrope"]["temperature_C"] < 94
        assert not chart["errors"], chart["errors"]
        assert len(chart["binodal"]["T"]) == 11
        with page.expect_download() as exported:
            page.click("#export-chart-svg")
        svg_path = REPORT / "butanol-water-vlle.svg"
        exported.value.save_as(svg_path)
        svg = svg_path.read_text()
        assert "VLLE · NRTL · 1 bar" in svg
        assert "Binodal · liquid 1" in svg and "Binodal · liquid 2" in svg
        assert "Temperature (°C)" in svg and "Phase mole fraction" not in svg
        page.screenshot(path=REPORT / "butanol-water-vlle.png", full_page=True)
        assert not failures, failures
        print(
            "PASS: real NRTL butanol/water constant-pressure VLLE envelope, binodal, data/SVG exports",
            flush=True,
        )
        print("Heteroazeotrope:", chart["heteroazeotrope"], flush=True)
        print("Artifacts:", REPORT, flush=True)
        browser.close()


if __name__ == "__main__":
    print("Binary VLLE browser report:", REPORT, flush=True)
    run()
