"""Opt-in root entry, real read-only previews, publication and review UI."""

import json
import os
import shutil
import sys
from unittest.mock import patch

from browser_web import REPORT, ROOT, preview
from playwright.sync_api import expect, sync_playwright


def check_fitted_review(page, address):
    # Exercise the moved review panel with a computed fit as well. Seed
    # observations through the public API so this check is independent of
    # the paper-table import workflow covered by other browser scripts.
    token = page.request.get(address + "/api/session").json()["csrf_token"]
    response = page.request.post(address + "/api/fitting", headers={"X-CSRF-Token": token}, data={
        "components": ["ethanol", "water"], "model": "NRTL", "form": "constant",
        "starts": 1, "max_nfev": 100, "cv": {"method": "none"},
        "observations": [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 3, "gamma2_inf": 2}],
    })
    assert response.status == 202, response.text()
    job_id = response.json()["job_id"]
    page.evaluate("""async id => {
        const {pollJob} = await import('/static/js/common.js');
        const job = await pollJob(id, () => {});
        if (job.status !== 'completed') throw new Error(job.error || job.status);
    }""", job_id)
    response = page.request.post(address + "/api/fitting/submit", headers={"X-CSRF-Token": token}, data={"job_id": job_id, "source": {"citation": "Synthetic computed-fit review check"}})
    assert response.status == 201, response.text()
    fit_id = response.json()["submission"]["id"]
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator("#fit-admin-refresh").click()
    row = page.locator("#fit-admin-list tbody tr").filter(has_text="Synthetic computed-fit review check")
    expect(row).to_have_count(1)
    row.get_by_role("button", name="Review", exact=True).click()
    page.locator("#modal").get_by_role("button", name="Approve", exact=True).click()
    expect(page.locator("#fit-admin-review-status")).to_contain_text("Approved and saved")
    expect(page.locator("#modal").get_by_role("button", name="Publish to runtime", exact=True)).to_be_visible()
    page.locator("#modal").get_by_role("link", name="View fit assessment", exact=True).click()
    expect(page.locator("#fit-result-status")).to_contain_text("Converged", timeout=15000)
    expect(page.locator("#fit-table tbody tr")).to_have_count(1)
    expect(page.locator("#fit-source")).to_have_value("Synthetic computed-fit review check")
    assert "review=" not in page.url
    assert page.locator("#fit-admin").count() == 0
    page.locator("#fit-admin-direct").click()
    expect(page.locator("#fit-submission-status")).to_contain_text("published:", timeout=120000)
    page.locator("#fit-publishing-link").click()
    expect(page.locator("#fit-admin-list")).to_contain_text("published")
    assert page.request.get(address + f"/api/fitting/admin/submissions/{fit_id}").json()["submission"]["status"] == "approved"


def main():
    runtime = REPORT / "runtime"
    runtime.mkdir()
    for model in ("nrtl", "uniquac", "eos"):
        (runtime / f"{model}_binary_interactions_cas.json").write_text(json.dumps({"metadata": {}, "interactions": []}))
    shutil.copyfile(ROOT / "data/uniquac_rq_cas.json", runtime / "uniquac_rq_cas.json")
    with patch.dict(os.environ, {"PFDSIM_INTERACTION_DATA_DIR": str(runtime)}), preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which("google-chrome"), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-publishing-link")).to_be_hidden()
        assert page.locator("#fit-admin").count() == 0
        page.locator("#account-button").click()
        page.locator("#modal").get_by_role("button", name="Create account", exact=True).click()
        page.locator('#modal input[name="username"]').fill("root")
        page.locator('#modal input[name="password"]').fill("root publishing browser password")
        page.locator('#modal input[name="setup_token"]').fill((REPORT / "jobs/root-setup-token").read_text().strip())
        page.locator('#modal form button[type="submit"]').click()
        expect(page.locator("#fit-publishing-link")).to_be_visible(timeout=15000)
        page.locator("#fit-publishing-link").click()
        expect(page.locator("#fit-admin")).to_be_visible()
        if "--review-only" in sys.argv:
            check_fitted_review(page, address)
            assert not errors, errors
            browser.close()
            print("Computed-fit review browser checks passed. Artifacts:", REPORT)
            return
        page.locator("#publish-value-12").fill("900")
        page.locator("#publish-value-21").fill("-200")
        page.locator("#publish-preview-points").fill("10")
        page.locator("#publish-preview-temperature").fill("330")
        page.locator("#publish-fit-method").fill("Published weighted least squares")
        page.locator("#publish-statistics").fill("RMS 1.2%; 42 data points")

        def check_preview(kind):
            page.locator("#publish-preview-kind").select_option(kind)
            page.locator("#publish-preview").click()
            expect(page.locator("#publish-preview-status")).to_contain_text("Preview complete", timeout=120000)
            expect(page.locator("#publish-preview-plots svg")).to_have_count(1)
            expect(page.locator("#publish-preview-parameters")).to_contain_text("INTERACTION_PARAMETERS")

        check_preview("GAMMA")
        page.locator("#publish-value-12").fill("1000")
        expect(page.locator("#publish-preview-status")).to_contain_text("Inputs changed")
        page.locator("#publish-value-21").fill("")
        page.reload()
        expect(page.locator("#publish-value-12")).to_have_value("1000")
        expect(page.locator("#publish-value-21")).to_have_value("")
        page.locator("#publish-value-21").fill("-200")
        expect(page.locator("#publish-statistics")).to_have_value("RMS 1.2%; 42 data points")
        check_preview("VLE")
        page.locator("#publish-basis").select_option("tau")
        page.locator("#publish-value-12").fill("4")
        page.locator("#publish-value-21").fill("4")
        check_preview("LLE")
        expect(page.locator("#fit-admin-list tbody tr")).to_have_count(0)
        assert json.loads((runtime / "nrtl_binary_interactions_cas.json").read_text())["interactions"] == []
        page.locator("#publish-model").select_option("UNIQUAC")
        expect(page.locator("#publish-equation")).to_contain_text("ln(τ)")
        page.locator("#publish-value-12").fill("0.6")
        page.locator("#publish-value-21").fill("1.2")
        check_preview("GAMMA")
        page.get_by_text("Exact preview parameters and PFD", exact=True).click()
        with page.expect_download() as downloaded:
            page.locator("#publish-preview-download").click()
        downloaded.value.save_as(REPORT / "manual-preview.pfd")
        assert "tau12_a" in (REPORT / "manual-preview.pfd").read_text()
        page.locator("#publish-source").fill("Browser-only synthetic source")
        page.locator("#publish-run").click()
        expect(page.locator("#fit-admin-status")).to_contain_text("published:", timeout=120000)
        expect(page.locator("#fit-admin-list tbody tr")).to_have_count(1)
        page.locator("#fit-admin-list").get_by_role("button", name="Review", exact=True).click()
        expect(page.locator("#modal")).to_contain_text("Source-reported fitting method: Published weighted least squares")
        expect(page.locator("#modal")).to_contain_text("RMS 1.2%; 42 data points")
        assert page.locator("#modal").get_by_role("link", name="View fit assessment").count() == 0
        with page.expect_download() as downloaded:
            page.locator("#modal").get_by_role("button", name="Download fitted PFD", exact=True).click()
        downloaded.value.save_as(REPORT / "manual-published.pfd")
        page.locator("#fit-admin-review-notes").fill("Synthetic browser check completed")
        page.locator("#modal").get_by_role("button", name="Withdraw and rebuild", exact=True).click()
        expect(page.locator("#fit-admin-status")).to_contain_text("revoked:", timeout=120000)
        payload = json.loads((runtime / "uniquac_binary_interactions_cas.json").read_text())
        assert not any(record.get("user_fit_id") for record in payload["interactions"])
        page.screenshot(path=REPORT / "parameter-publishing-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=REPORT / "parameter-publishing-mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        check_fitted_review(page, address)
        assert not errors, errors
        browser.close()
    print("Parameter publishing browser checks passed. Artifacts:", REPORT)


if __name__ == "__main__":
    main()
