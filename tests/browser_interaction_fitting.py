"""Opt-in Chrome smoke coverage of the complete fitting workflow.

Run: python tests/browser_interaction_fitting.py
Uses the same isolated preview lifecycle as browser_web.py. No external data.
"""

import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_web import REPORT, preview
from tests.test_interaction_fitting import synthetic
from tests.fitting_import_samples import WIKIPEDIA_TXY, UNLABELED_TXY
from scripts import build_cas_interaction_parameters as builder


def main():
    data, _, _ = synthetic(kinds=("GAMMA_INF", "HE"))
    runtime = REPORT / "runtime"
    runtime.mkdir()
    # Build isolated reference tables without importing the live review store.
    # Copying published tables would leave foreign fit IDs without provenance
    # and correctly prevent this preview's administrator from rebuilding them.
    resolved, unresolved = builder.resolve_component_ids()
    for prefix in ("nrtl", "uniquac", "eos"):
        builder.write_json(
            runtime / f"{prefix}_binary_interactions_cas.json",
            builder.build_interaction_payload(
                f"{prefix}_binary_interactions.json",
                resolved,
                unresolved,
                user_fits_path=REPORT / "empty-user-fits.sqlite",
            ),
        )
    shutil.copyfile(
        builder.DATA / "uniquac_rq_cas.json", runtime / "uniquac_rq_cas.json"
    )
    with (
        patch.dict(os.environ, {"PFDSIM_INTERACTION_DATA_DIR": str(runtime)}),
        preview() as address,
        sync_playwright() as playwright,
    ):
        executable = shutil.which("google-chrome")
        if executable is None:
            raise RuntimeError(
                "Install Google Chrome to run this opt-in browser check."
            )
        browser = playwright.chromium.launch(
            executable_path=executable, args=["--no-sandbox"]
        )
        context = browser.new_context(viewport={"width": 1500, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def report_failed_response(response):
            if response.status >= 400 and "/api/" in response.url:
                print(response.url, response.status, response.text(), flush=True)

        page.on("response", report_failed_response)
        page.goto(address + "/parameter-fitting")
        expect(page.locator("#fit-vapor option")).to_have_count(8)
        expect(page.locator(".fit-data-toolbar #fit-data-kind")).to_be_visible()
        page.locator("#fit-help").click()
        expect(page.locator("#modal")).to_contain_text("Validation-only observations")
        page.locator("#modal-close").click()
        page.locator("#fit-model").select_option("UNIQUAC")
        page.locator("#fit-prefill").click()
        expect(page.locator("#fit-progress")).to_have_text(
            "R/Q prefilled.", timeout=120000
        )
        assert float(page.locator("#fit-r1").input_value()) > 0
        page.locator("#fit-model").select_option("NRTL")
        page.locator("#fit-input").fill(
            "| kind | T_C | x1 | HE_J_mol |\n| --- | --- | --- | --- |\n| HE | 25 | 0.5 | 200 |"
        )
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        page.locator("#fit-input").fill(WIKIPEDIA_TXY)
        page.locator("#fit-parse").click()
        expect(page.locator("#modal")).to_be_visible()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        page.locator("#modal").get_by_role("button", name="Cancel", exact=True).click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(1)
        page.locator("#fit-parse").click()
        page.get_by_role("button", name="Use acetone as component 1").click()
        page.locator("#fit-import-pressure").fill("1")
        page.locator("#fit-import-pressure_unit").select_option("atm")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(17)
        assert (
            page.locator("#fit-table tbody tr")
            .nth(1)
            .locator("td")
            .first.text_content()
            == "2"
        )
        page.locator("#fit-input").fill(UNLABELED_TXY)
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-import-column-0")).to_have_value("temperature")
        page.locator("#fit-import-pressure").fill("1")
        page.locator("#fit-import-pressure_unit").select_option("atm")
        expect(page.locator("#fit-import-add")).to_be_enabled()
        page.locator("#fit-import-add").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(33)
        page.locator("#fit-set-uncertainty").click()
        page.locator("#fit-sigma-HE_J_mol").fill("25")
        page.locator("#fit-sigma-target").select_option("set-1")
        page.locator("#fit-apply-sigma").click()
        expect(page.locator('input[aria-label="Observation 1 sigma"]')).to_have_value(
            '{"HE_J_mol":25}'
        )
        page.locator("#modal").get_by_role("button",name="Done",exact=True).click()
        page.locator("#fit-input-method").select_option("manual")
        page.locator("#fit-data-kind").select_option("HE")
        page.locator("#fit-manual-temperature").fill("25")
        page.locator("#fit-manual-x1").fill("0.5")
        page.locator("#fit-manual-enthalpy").fill("200")
        page.locator("#fit-manual-validation-only").check()
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(34)
        expect(page.locator('input[aria-label="Observation 34 sigma"]')).to_have_value(
            '{"HE_J_mol":25}'
        )
        page.locator("#fit-model").select_option("UNIQUAC")
        page.locator("#fit-reset").click()
        expect(page.locator("#fit-model")).to_have_value("NRTL")
        expect(page.locator("#fit-table tbody tr")).to_have_count(34)
        expect(page.locator("#fit-comp1")).to_have_value("acetone")
        page.reload()
        expect(page.locator("#fit-table tbody tr")).to_have_count(34)
        page.locator("#fit-clear").click()
        page.locator("#fit-clear-kind").select_option("all")
        page.locator("#fit-clear-confirm").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(0)
        page.locator("#fit-input-method").select_option("paste")
        page.locator("#fit-comp1").fill("ethanol")
        page.locator("#fit-set-uncertainty").click()
        page.locator("#fit-sigma-HE_J_mol").fill("")
        page.locator("#modal").get_by_role("button",name="Done",exact=True).click()
        page.locator("#fit-input").fill(json.dumps(data["observations"]))
        page.locator("#fit-parse").click()
        expect(page.locator("#fit-table tbody tr")).to_have_count(
            len(data["observations"])
        )
        page.locator('input[aria-label="Validation-only observation 1"]').check()
        page.get_by_text("Advanced controls", exact=True).click()
        page.locator("#fit-starts").fill("1")
        page.get_by_text("Cross-validation", exact=True).click()
        page.locator("#fit-cv").select_option("kfold")
        page.locator("#fit-folds").fill("2")
        page.locator("#fit-run").click()
        expect(page.locator("#fit-result-status")).to_contain_text(
            "Converged", timeout=120000
        )
        expect(page.locator("#fit-validation tbody tr")).to_have_count(2)
        expect(page.locator("#fit-physical-metrics")).to_contain_text("RMSE")
        expect(page.locator("#fit-physical-metrics")).to_contain_text("Validation-only")
        expect(page.locator("#fit-objectives")).not_to_be_visible()
        assert page.locator(".fit-objective-card svg").count()>=3
        expect(page.locator("#fit-entry")).to_have_value(
            __import__("re").compile("INTERACTION_PARAMETERS:")
        )
        with page.expect_download() as received:
            page.locator("#fit-download").click()
        received.value.save_as(REPORT / "fitted-mixture.pfd")
        original = "PROCESS: Browser export\nONLINE_LOOKUP: false\nCOMPONENTS:\n    A | ethanol\n    B | water\n"
        source = REPORT / "original.pfd"
        source.write_text(original)
        page.locator("#fit-export-file").set_input_files(source)
        expect(page.locator("#fit-map-0")).to_be_visible()
        with page.expect_download() as received:
            page.locator("#fit-merged-download").click()
        received.value.save_as(REPORT / "updated-process.pfd")
        assert "A/B | model=NRTL" in (REPORT / "updated-process.pfd").read_text()
        project_id = "fitting-browser-laboratory"
        page.evaluate(
            """async ({text, id}) => {
          const {api, writeLocal} = await import('/static/js/common.js');
          const parsed = await api('/api/parse', {text});
          writeLocal('pfdsim.laboratories.v1', {[id]: {id, text: parsed.text, pfd: parsed.pfd, pending: false, filename: 'process.pfd'}});
          writeLocal('pfdsim.last.v1', id);
        }""",
            {"text": original, "id": project_id},
        )
        workspace = page.context.new_page()
        workspace.goto(address + "/editor")
        expect(workspace.locator('[data-equipment="Pump"]')).to_be_visible()
        page.reload()
        expect(page.locator("#fit-result-status")).to_contain_text("Converged")
        page.locator("#fit-export-project").select_option(project_id)
        page.locator("#fit-apply").click()
        expect(page.locator("#notifications")).to_contain_text("Fit applied")
        applied = page.evaluate(
            "id => JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id]",
            project_id,
        )
        assert applied["pfd"]["metadata"]["thermo_method"] == "NRTL"
        workspace.locator('[data-equipment="Pump"]').click()
        expect(workspace.locator("#notifications")).to_contain_text(
            "changed in another tab"
        )
        assert "A/B | model=NRTL" in page.evaluate(
            "id => JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id].text",
            project_id,
        )
        workspace.close()
        page.locator("#fit-source").fill(
            "Synthetic browser validation; not experimental reference data"
        )
        page.locator("#fit-submit").click()
        expect(page.locator("#fit-submission-status")).to_contain_text(
            "Submitted for review"
        )
        # Claim root through the real account dialog and publish to this
        # preview's isolated runtime directory, never the repository tables.
        page.locator("#account-button").click()
        page.locator("#modal").get_by_role(
            "button", name="Create account", exact=True
        ).click()
        page.locator('#modal input[name="username"]').fill("root")
        page.locator('#modal input[name="password"]').fill(
            "browser admin long password"
        )
        setup_token = (REPORT / "jobs/root-setup-token").read_text().strip()
        page.locator('#modal input[name="setup_token"]').fill(setup_token)
        page.locator('#modal form button[type="submit"]').click()
        expect(page.locator("#fit-admin")).to_be_visible(timeout=15000)
        expect(page.locator("#fit-admin-list tbody tr")).to_have_count(1)
        page.locator("#fit-admin-list").get_by_role(
            "button", name="Review", exact=True
        ).click()
        # Approval must work without a mandatory note and remain visible with
        # its publication/export actions in the same open review dialog.
        page.locator("#modal").get_by_role("button", name="Approve", exact=True).click()
        expect(page.locator("#fit-admin-list")).to_contain_text("approved")
        expect(page.locator("#modal")).to_be_visible()
        expect(page.locator("#fit-admin-review-status")).to_contain_text("Approved and saved")
        expect(page.locator("#modal").get_by_role("button",name="Publish to runtime",exact=True)).to_be_visible()
        expect(page.locator("#modal").get_by_role("button",name="Download fit result",exact=True)).to_be_visible()
        expect(page.locator("#fit-admin-list")).to_contain_text("Ethanol")
        page.locator("#modal-close").click()
        page.locator("#fit-admin-direct").click()
        expect(page.locator("#fit-admin-status")).to_contain_text(
            "published:", timeout=120000
        )
        assert (
            json.loads((runtime / "nrtl_binary_interactions_cas.json").read_text())[
                "metadata"
            ]["user_activity_fit_records"]
            == 1
        )
        page.screenshot(path=REPORT / "fitting-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=REPORT / "fitting-mobile.png", full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.reload()
        expect(page.locator("#fit-result-status")).to_contain_text("Converged")
        assert not errors, errors
        browser.close()
    print(f"Fitting browser workflow passed. Artifacts: {REPORT}")


if __name__ == "__main__":
    main()
