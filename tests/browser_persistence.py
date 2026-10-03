"""Browser regressions for empty drafts and concurrent account autosaves.

Run with Playwright and Chrome: python tests/browser_persistence.py
Account responses are mocked; no calculation jobs are started.
"""

import os
import shutil
import tempfile

from browser_web import REPORT, preview
from playwright.sync_api import expect, sync_playwright


def run():
    os.environ["PFDSIM_WEB_DATA"] = tempfile.mkdtemp(prefix="pfdsim-persistence-web-")
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            args=["--no-sandbox"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.goto(address + "/editor")
        expect(page.locator(".palette-item")).to_have_count(33)
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill("UNIT Kept : Pump\n")
        page.click("#apply-source")
        expect(page.locator("#source-status")).to_have_text("Source synchronized")
        page.locator("#code-editor").fill("")
        page.reload()
        expect(page.locator(".palette-item")).to_have_count(33)
        page.locator("[data-view=code]").click()
        expect(page.locator("#code-editor")).to_have_value("")
        expect(page.locator("#source-status")).to_have_text("Unapplied text draft")
        expect(page.locator("[data-unit=Kept]")).to_have_count(1)
        context.close()
        print("PASS: empty source drafts retain the applied flowsheet after reload", flush=True)

        context = browser.new_context()
        definition = context.request.get(address + "/api/config").json()["empty_pfd"]
        identifier = "concurrent-laboratory"
        remote = {
            "id": identifier,
            "version": 1,
            "updated": 1,
            "document": {
                "pfd": definition,
                "text": "PROCESS: Original\n",
                "pending": False,
                "filename": "process.pfd",
                "job": None,
                "lastJob": None,
                "layout": None,
            },
        }
        submissions = []
        context.route(
            "**/api/session",
            lambda route: route.fulfill(json={
                "success": True,
                "csrf_token": "review-token",
                "user": {"id": "review-account", "username": "Review"},
                "quota": {"remaining_seconds": 900, "limit_seconds": 900},
            }),
        )
        context.route(
            "**/api/flowsheets",
            lambda route: route.fulfill(json={"success": True, "flowsheets": [remote]}),
        )

        def save(route):
            payload = route.request.post_data_json
            submissions.append(payload)
            if payload["version"] != remote["version"]:
                route.fulfill(status=409, json={"error": "Another session saved this laboratory."})
            else:
                remote.update(version=remote["version"] + 1, document=payload["document"])
                route.fulfill(json={"success": True, "version": remote["version"], "updated": 2})

        context.route("**/api/flowsheets/" + identifier, save)
        first, second = context.new_page(), context.new_page()
        for tab in (first, second):
            tab.on("pageerror", lambda error: failures.append(str(error)))
            tab.goto(address + "/vle-chart")
            expect(tab.locator("#chart-method")).to_have_value("UNIFAC")
        first.evaluate("""async id => {
            const module = await import('/static/js/persistence.js');
            const key = 'pfdsim.laboratories.v1';
            const library = JSON.parse(localStorage.getItem(key));
            library[id].text = 'PROCESS: First tab edit\\n';
            localStorage.setItem(key, JSON.stringify(library));
            module.queueCloud(library[id]);
        }""", identifier)
        first.wait_for_function("""id => JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id].cloudVersion === 2""", arg=identifier)
        second.evaluate("""async id => {
            const module = await import('/static/js/persistence.js');
            const key = 'pfdsim.laboratories.v1';
            const library = JSON.parse(localStorage.getItem(key));
            // The second tab still edits the original revision, although the
            // shared browser library now carries the first tab's save metadata.
            library[id].text = 'PROCESS: Second tab edit\\n';
            localStorage.setItem(key, JSON.stringify(library));
            module.queueCloud(library[id]);
        }""", identifier)
        second.wait_for_function("""id => !!JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id].cloudConflict""", arg=identifier)
        assert [item["version"] for item in submissions] == [1, 1], submissions
        assert remote["document"]["text"] == "PROCESS: First tab edit\n"
        saved = second.evaluate("""id => JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id]""", identifier)
        assert saved["text"] == "PROCESS: Second tab edit\n"
        assert saved["cloudConflict"]["text"] == "PROCESS: First tab edit\n"
        context.close()
        print("PASS: concurrent tabs preserve both edits and report an account-save conflict", flush=True)

        context = browser.new_context(viewport={"width": 390, "height": 844})
        context.add_init_script("""localStorage.setItem('pfdsim.settings.v1', JSON.stringify({theme:'light'}));""")
        page = context.new_page()
        page.on("pageerror", lambda error: failures.append(str(error)))
        chart = {
            "chart_type": "PXY", "method": "IDEAL", "temperature_C": 25,
            "comp1_name": "Ethanol", "comp2_name": "Water",
            "x": [0, 0.5, 1], "y": [0, 0.7, 1],
            "P_bubble": [0.03, 0.055, 0.08], "P_dew": [0.03, 0.055, 0.08],
            "errors": [],
        }
        page.route("**/api/vle-chart", lambda route: route.fulfill(status=202, json={"job_id": "contrast-review"}))
        page.route("**/api/jobs/contrast-review", lambda route: route.fulfill(json={
            "job": {"status": "completed", "output": chart, "progress": []},
        }))
        page.goto(address + "/vle-chart")
        expect(page.locator("#chart-method")).to_have_value("UNIFAC")
        expect(page.locator('#chart-method option[value="STEAM"]')).to_have_attribute("disabled", "")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text("Diagram complete.")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        contrast = page.evaluate("""async () => {
            const {toast} = await import('/static/js/common.js');
            toast('Saved successfully');
            toast('Unable to save', true);
            function luminance(color) {
                const channels = color.match(/[\\d.]+/g).slice(0, 3).map(Number).map(v => v / 255);
                return channels.map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4)
                    .reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
            }
            function ratio(foreground, background) {
                const a = luminance(foreground), b = luminance(background);
                return (Math.max(a,b) + .05) / (Math.min(a,b) + .05);
            }
            const background = getComputedStyle(document.getElementById('chart-frame')).backgroundColor;
            return {
                text: ratio(getComputedStyle(document.querySelector('.chart-text')).fill, background),
                curves: [...document.querySelectorAll('.chart-line')].map(node => ratio(getComputedStyle(node).stroke, background)),
                toasts: [...document.querySelectorAll('.toast')].map(node => {
                    const style = getComputedStyle(node);
                    return ratio(style.color, style.backgroundColor);
                }),
            };
        }""")
        assert contrast["text"] >= 4.5, contrast
        assert min(contrast["curves"]) >= 3, contrast
        assert min(contrast["toasts"]) >= 4.5, contrast
        assert page.locator(".chart-axis").evaluate("node => getComputedStyle(node).fill") == "none"
        with page.expect_download() as exported:
            page.click("#export-chart-svg")
        artifact = REPORT / "light-phase-diagram.svg"
        exported.value.save_as(artifact)
        figure = artifact.read_text()
        assert "Ethanol / Water" in figure
        assert "PXY · IDEAL · 25 °C" in figure
        assert "Bubble curve · liquid x" in figure
        assert "Dew curve · vapor y" in figure
        assert ".chart-axis{fill:none;" in figure
        page.screenshot(path=REPORT / "light-mobile-atlas.png", full_page=True)
        assert not failures, failures
        browser.close()
        print("PASS: light-theme chart/text/toast contrast and mobile atlas width", contrast, flush=True)
        print("Browser artifacts:", REPORT)


if __name__ == "__main__":
    print("Persistence browser report:", REPORT, flush=True)
    run()
