"""Opt-in real-browser coverage for the laboratory and phase atlas.

Run with a Python environment containing Playwright and an installed Chrome:
    python tests/browser_web.py
The server uses the repository virtualenv when present. Reports/screenshots
are written under /tmp, and this script owns and stops its preview processes.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(tempfile.mkdtemp(prefix="pfdsim-browser-"))


@contextmanager
def preview():
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    job_directory = REPORT / "jobs"
    script = """import sys
from pathlib import Path
from app import app
app.config['JOB_DIRECTORY']=Path(sys.argv[1])
app.run(host='127.0.0.1',port=int(sys.argv[2]))
"""
    if "--gunicorn" in sys.argv:
        script = """import sys
from pathlib import Path
from app import app
from gunicorn.app.base import BaseApplication
app.config['JOB_DIRECTORY']=Path(sys.argv[1])
class Preview(BaseApplication):
    def load_config(self):
        for key,value in {'bind':'127.0.0.1:'+sys.argv[2],'workers':2,'threads':4,'worker_class':'gthread','timeout':60}.items():
            self.cfg.set(key,value)
    def load(self):
        return app
Preview().run()
"""
    with (REPORT / "server.log").open("w") as log:
        child = subprocess.Popen(
            [str(python), "-c", script, str(job_directory), str(port)],
            cwd=ROOT,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    address = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError((REPORT / "server.log").read_text())
            try:
                with urlopen(address + "/api/config", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Preview did not start")
        yield address
    finally:
        # Only this preview and its detached calculation workers are stopped.
        database = job_directory / "web.sqlite"
        if database.exists():
            import sqlite3

            with sqlite3.connect(database) as db:
                worker_pids = [
                    row[0]
                    for row in db.execute(
                        "SELECT pid FROM workers WHERE pid IS NOT NULL"
                    )
                ]
            for pid in worker_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def label_input(container, label):
    return (
        container.locator("label")
        .filter(
            has=container.page.locator(
                ".field-label", has_text=re.compile("^" + re.escape(label) + "$")
            )
        )
        .locator("input,select,textarea")
        .first
    )


def run():
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise RuntimeError("Install Chrome or Chromium for this browser check")
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=chrome, headless=True, args=["--no-sandbox"]
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000}, accept_downloads=True
        )
        page = context.new_page()
        failures = []
        page.on(
            "pageerror",
            lambda error: (
                failures.append(str(error)),
                print("PAGE ERROR:", error, flush=True),
            ),
        )

        def request_error(response):
            if "/api/" in response.url and response.status >= 400:
                print(
                    "API ERROR:",
                    response.status,
                    response.url,
                    response.text(),
                    flush=True,
                )
                if response.url.endswith("/api/layout"):
                    print(
                        "Layout positions:",
                        [
                            (u["id"], u.get("x"), u.get("y"))
                            for u in response.request.post_data_json["pfd"]["units"]
                        ],
                        flush=True,
                    )

        page.on("response", request_error)
        page.goto(address)
        expect(page.locator(".palette-item")).to_have_count(33)
        page.screenshot(path=REPORT / "title.png", full_page=True)
        page.click("#continue-project")
        page.locator('[data-equipment="Pump"]').click()
        expect(page.locator("[data-unit]")).to_have_count(1)
        before = page.locator('[data-unit="P-1"]').bounding_box()
        page.mouse.move(before["x"] + 90, before["y"] + 25)
        page.mouse.down()
        page.mouse.move(before["x"] - 120, before["y"] - 30, steps=8)
        page.mouse.up()
        expect(page.locator("#workspace-status")).to_have_text(
            "Ready to explore", timeout=15000
        )
        page.locator('[data-equipment="Heater"]').click()
        expect(page.locator("[data-unit]")).to_have_count(2)
        page.locator('[data-unit="P-1"] .port.outlet').click()
        page.locator('[data-unit="H-1"] .port.inlet').click()
        expect(page.locator("[data-stream]")).to_have_count(1)
        page.click("#undo-button")
        expect(page.locator("[data-stream]")).to_have_count(0)
        page.click("#redo-button")
        expect(page.locator("[data-stream]")).to_have_count(1)
        page.click("#add-stream-button")
        page.locator("#modal select").nth(1).select_option("P-1.in")
        page.get_by_role("button", name="Create stream", exact=True).click()
        print(
            "After feed creation:",
            page.locator("#notifications").inner_text(),
            flush=True,
        )
        expect(page.locator("[data-stream]")).to_have_count(2)
        page.click("#add-stream-button")
        page.locator("#modal select").nth(0).select_option("H-1.out")
        page.get_by_role("button", name="Create stream", exact=True).click()
        expect(page.locator("[data-stream]")).to_have_count(3)
        page.click("#components-button")
        page.locator("#modal").get_by_role(
            "button", name="＋ Add entry", exact=True
        ).click()
        entry = (
            page.locator("#modal .structured-group")
            .filter(
                has=page.locator(":scope > summary", has_text=re.compile("^Entry$"))
            )
            .first
        )
        label_input(entry, "Flowsheet symbol").fill("H2O")
        label_input(entry, "Chemical identifier (name, formula, CAS or SMILES)").fill(
            "Water"
        )
        # Optional fields and nested component tabs remain accessible.
        entry.get_by_role("button", name="Solids", exact=True).click()
        entry.locator(":scope > .add-field select").select_option("rho_solid")
        entry.locator(":scope > .add-field button").click()
        expect(entry.get_by_role("button", name="Solids", exact=True)).to_have_class(
            "active"
        )
        label_input(entry, "Solid density (kg/m³)").fill("917")
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()
        # Configure feed composition with structured fields, not source code.
        page.locator('[data-stream="FEED-1"]').press("Enter")
        page.locator("#inspector-tabs").get_by_role(
            "button", name="Composition", exact=True
        ).click()
        page.screenshot(path=REPORT / "composition-controls.png", full_page=True)
        label_input(page.locator("#inspector-content"), "H2O · Water").fill("1")
        page.locator("#inspector-content").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        # Source drafts survive tab switches and browser reloads.
        page.locator('[data-view="code"]').click()
        original = page.locator("#code-editor").input_value()
        draft = original + "\nTHIS IS INVALID\n"
        page.locator("#code-editor").fill(draft)
        page.locator('[data-view="diagram"]').click()
        page.locator('[data-view="code"]').click()
        assert page.locator("#code-editor").input_value() == draft
        page.reload()
        page.locator('[data-view="code"]').click()
        expect(page.locator("#code-editor")).to_have_value(draft)
        page.click("#apply-source")
        expect(page.locator("#source-status")).not_to_have_text("Source synchronized")
        expect(page.locator("[data-unit]")).to_have_count(2)
        page.click("#discard-source")
        page.get_by_role("button", name="Discard text", exact=True).click()
        expect(page.locator("#source-status")).to_have_text("Source synchronized")
        page.locator('[data-view="diagram"]').click()
        page.screenshot(path=REPORT / "editor.png", full_page=True)
        # Load and run a real example, including repeat reuse and export.
        page.click("#examples-button")
        page.locator("#modal input[type=search]").fill("simple_flash.pfd")
        page.locator(".example-item").click()
        expect(page.locator("#project-name")).to_have_text("Simple Flash Separation")
        page.click("#fit-button")
        page.screenshot(path=REPORT / "example.png", full_page=True)
        page.click("#run-button")
        expect(page.locator("#run-status")).to_have_text(
            "Experiment complete · converged", timeout=180000
        )
        expect(page.locator("#download-results")).to_be_enabled()
        with page.expect_download() as exported:
            page.click("#download-results")
        assert exported.value.suggested_filename.endswith(".pfr")
        page.click("#run-button")
        expect(page.locator("#run-status")).to_have_text(
            "Experiment complete · converged", timeout=180000
        )
        expect(page.locator("#run-progress")).to_contain_text(
            "Reusing initialized flowsheet"
        )
        expect(page.locator("#result-content pre")).to_have_count(0)
        page.locator("#result-content .tabs").get_by_role(
            "button", name=re.compile("^Equipment")
        ).click()
        page.locator("#result-content").get_by_role(
            "button", name="HEAT-1", exact=True
        ).click()
        expect(page.locator("#result-content")).to_contain_text("Heat duty (kW)")
        page.screenshot(path=REPORT / "results.png", full_page=True)
        # Large examples use shared engineering placement and orthogonal routes.
        page.click("#examples-button")
        page.locator("#modal input[type=search]").fill("haber_bosch_full.pfd")
        page.locator(".example-item").click()
        expect(page.locator("[data-unit]")).to_have_count(35, timeout=15000)
        page.click("#fit-button")
        page.screenshot(path=REPORT / "large-flowsheet.png", full_page=True)
        page.locator('[data-unit="T-101"]').press("Enter")
        expect(
            page.locator('#inspector-content [data-setting="n_stages"]')
        ).to_be_visible()
        expect(
            page.locator('#inspector-content [data-setting="reflux_ratio"]')
        ).to_be_visible()
        expect(
            page.locator(
                '#inspector-content [data-setting="n_stages"] input[type=text]'
            )
        ).to_have_value("30")
        page.screenshot(path=REPORT / "guided-distillation.png", full_page=True)
        page.click("#examples-button")
        page.locator("#modal input[type=search]").fill("simple_flash.pfd")
        page.locator(".example-item").click()
        expect(page.locator("#project-name")).to_have_text(
            "Simple Flash Separation", timeout=15000
        )
        page.get_by_role("link", name="Settings", exact=True).click()
        label_input(page.locator("#settings-content"), "Theme").select_option("light")
        page.get_by_role("button", name="Save settings", exact=True).click()
        expect(page.locator("body")).to_have_attribute("data-theme", "light")
        page.screenshot(path=REPORT / "settings.png", full_page=True)
        page.get_by_role("link", name="Phase atlas", exact=True).click()
        expected_methods = len(
            page.request.get(address + "/api/config").json()["chart_methods"]
        )
        expect(page.locator("#chart-method option")).to_have_count(expected_methods)
        assert page.locator("#comp1").get_attribute("list") is None
        assert page.locator("#comp2").get_attribute("list") is None
        page.locator("#comp1").fill("64-17-5")
        page.locator("#comp2").fill("O")
        page.locator("#reference-browser > summary").click()
        page.locator("#reference-search").fill("water")
        expect(page.locator("#reference-chemical option").first).to_contain_text(
            "Perry", timeout=15000
        )
        page.locator("[data-reference-target=comp2]").click()
        expect(page.locator("#comp1")).to_have_value("64-17-5")
        page.locator("#chart-method").select_option("IDEAL")
        page.locator("#chart-type").select_option("PXY")
        page.get_by_text("Advanced options", exact=True).click()
        page.locator("#chart-points").fill("10")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text(
            "Diagram complete.", timeout=180000
        )
        expect(page.locator("#chart-frame svg")).to_have_count(1)
        with page.expect_download() as svg_download:
            page.click("#export-chart-svg")
        assert svg_download.value.suggested_filename.endswith(".svg")
        # A saved laboratory supplies symbols/properties to the phase tool.
        options = page.locator("#chart-source option").evaluate_all(
            "(options)=>options.map(o=>({value:o.value,text:o.textContent}))"
        )
        saved = next(
            option for option in options if option["text"] == "Simple Flash Separation"
        )
        page.locator("#chart-source").select_option(saved["value"])
        page.locator("#comp1").fill("C2H5OH")
        page.locator("#comp2").fill("H2O")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text(
            "Diagram complete.", timeout=180000
        )
        page.screenshot(path=REPORT / "binary-atlas.png", full_page=True)
        page.locator("#chart-source").select_option("independent")
        page.locator("#chart-method").select_option("NRTL")
        page.locator("#chart-type").select_option("TERNARY_LLE")
        page.locator("#comp1").fill("water")
        page.locator("#comp2").fill("methanol")
        page.locator("#comp3").fill("benzene")
        page.locator("#chart-points").fill("6")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text(
            "Diagram complete.", timeout=180000
        )
        expect(page.locator("#chart-frame svg circle[role=button]")).to_have_count(28)
        page.screenshot(path=REPORT / "ternary-lle.png", full_page=True)
        page.locator("#chart-type").select_option("TERNARY_VLLE")
        page.locator("#chart-points").fill("4")
        page.locator("#chart-temperature").fill("59.85")
        page.click("#generate-chart")
        expect(page.locator("#chart-status")).to_have_text(
            "Diagram complete.", timeout=180000
        )
        page.screenshot(path=REPORT / "ternary-vlle.png", full_page=True)
        # Responsive views and locally persistent settings.
        page.goto(address + "/settings")
        expect(page.locator("body")).to_have_attribute("data-theme", "light")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=REPORT / "mobile-settings.png", full_page=True)
        page.goto(address + "/editor")
        expect(page.locator(".palette-item")).to_have_count(33)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=REPORT / "mobile-editor.png", full_page=True)
        # Account autosaves include source drafts and recover in another browser.
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.click("#account-button")
        page.locator("#modal").get_by_role(
            "button", name="Create account", exact=True
        ).click()
        page.locator("#modal input[name=username]").fill("BrowserLab")
        page.locator("#modal input[name=password]").fill("browser laboratory password")
        page.locator("#modal button[type=submit]").click()
        expect(page.locator("#account-button")).to_have_text(
            "BrowserLab", timeout=15000
        )
        expect(page.locator("#cloud-save-status")).to_have_text(
            "Saved to your account", timeout=15000
        )
        page.locator("[data-view=code]").click()
        cloud_draft = page.locator("#code-editor").input_value() + "\n# account draft\n"
        page.locator("#code-editor").fill(cloud_draft)
        expect(page.locator("#cloud-save-status")).to_have_text(
            "Saved to your account", timeout=15000
        )
        # Wait for this exact revision, rather than a status from the last save.
        page.wait_for_function(
            "async ()=>{const r=await fetch('/api/flowsheets');const d=await r.json();return d.flowsheets.some(f=>f.document.text.endsWith('# account draft\\n'));}",
            timeout=15000,
        )
        second_context = browser.new_context(viewport={"width": 1440, "height": 1000})
        second_page = second_context.new_page()
        second_page.goto(address + "/editor")
        second_page.click("#account-button")
        second_page.locator("#modal input[name=username]").fill("BrowserLab")
        second_page.locator("#modal input[name=password]").fill(
            "browser laboratory password"
        )
        second_page.locator("#modal button[type=submit]").click()
        expect(second_page.locator("#account-button")).to_have_text(
            "BrowserLab", timeout=15000
        )
        second_page.get_by_role("link", name="PFDSIM title screen").click()
        second_page.click("#title-library")
        second_page.locator(".example-item").filter(
            has_text="Simple Flash Separation"
        ).first.click()
        second_page.locator("[data-view=code]").click()
        expect(second_page.locator("#code-editor")).to_have_value(cloud_draft)
        second_context.close()
        assert not failures, failures
        browser.close()
        print(
            "PASS: title, equipment, connections, undo/redo, source drafts, advanced forms, real simulation/reuse/export, settings, binary/ternary LLE/VLLE atlas, responsive layouts, account registration/login, and account autosave recovery"
        )
        print(f"Reports and screenshots: {REPORT}")


if __name__ == "__main__":
    print(f"Browser report directory: {REPORT}", flush=True)
    run()
