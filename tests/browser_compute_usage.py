"""Opt-in browser checks for the quota dropdown and fair-usage information."""

import os
import shutil
import time

from playwright.sync_api import sync_playwright, expect

from browser_web import REPORT, preview


def run():
    os.environ["PFDSIM_WEB_DATA"] = str(REPORT / "web-data")
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(10000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/settings")
        expect(page.locator("#compute-budget")).to_have_text("5m 0s CPU left")
        toggle = page.locator("#compute-budget-toggle")
        panel = page.locator("#compute-budget-panel")
        toggle.focus()
        page.keyboard.press("Enter")
        expect(panel).to_be_visible()
        expect(page.locator("#compute-usage-list")).to_contain_text("No recent tasks")
        expect(page.locator("#modal")).not_to_be_visible()
        page.keyboard.press("Escape")
        expect(panel).not_to_be_visible()
        expect(toggle).to_be_focused()

        quota = page.request.get(address + "/api/session").json()["quota"]
        page.route("**/api/usage", lambda route: route.fulfill(json={
            "success": True, "quota": quota,
            "recent_jobs": [
                {"kind": "simulation", "status": "completed", "created": time.time(), "cpu_seconds": 12.5},
                {"kind": "fit", "status": "failed", "created": time.time() - 60, "cpu_seconds": 4.1},
            ],
        }))
        for width, height, name in [(1440, 900, "desktop"), (390, 844, "mobile")]:
            page.set_viewport_size({"width": width, "height": height})
            toggle.click()
            expect(panel).to_be_visible()
            expect(page.locator("#compute-usage-list")).to_contain_text("12.5 CPU s")
            expect(page.locator("#compute-usage-list")).to_contain_text("Parameter fitting")
            expect(page.locator("#modal")).not_to_be_visible()
            bounds = panel.bounding_box()
            assert 0 <= bounds["x"] and bounds["x"] + bounds["width"] <= width
            assert 0 <= bounds["y"] and bounds["y"] + bounds["height"] <= height
            page.screenshot(path=REPORT / f"compute-usage-{name}.png", full_page=True)
            page.locator(".masthead").click(position={"x": 2, "y": 2})
            expect(panel).not_to_be_visible()

        toggle.click()
        page.locator("#compute-more-info").click()
        expect(panel).not_to_be_visible()
        expect(page.locator("#modal-title")).to_have_text("Usage and rate limits")
        expect(page.locator("#modal-content")).to_contain_text("PFDSim is free software")
        expect(page.locator("#modal-content")).to_contain_text("won't charge for subscriptions")
        expect(page.locator("#modal-content")).to_contain_text("fair usage for everyone")
        expect(page.locator("#modal a")).to_have_attribute("href", "mailto:pfdsim@chemicalprocess.org")
        page.keyboard.press("Escape")
        page.locator("#account-button").click()
        page.locator("#modal .tabs").get_by_role("button", name="Create account", exact=True).click()
        expect(page.locator("#modal-content")).not_to_contain_text("CPU minutes")
        expect(page.locator("#modal-content")).not_to_contain_text("fair usage")
        assert not errors, errors
        browser.close()
    print(f"Quota dropdown browser checks passed. Screenshots: {REPORT}")


if __name__ == "__main__":
    run()
