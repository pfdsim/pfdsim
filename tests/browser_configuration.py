"""Focused browser checks for friendly physical configuration workflows."""

import re
import shutil

from playwright.sync_api import sync_playwright, expect
from browser_web import preview, label_input, REPORT

SOURCE = """PROCESS: Configuration controls
THERMO_METHOD: NRTL
ONLINE_LOOKUP: false
COMPONENTS:
    H2 | Hydrogen
    O2 | Oxygen
    H2O | Water
STREAM Feed : FEED -> R-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2:0.6, O2:0.4
STREAM Product : R-1.out -> PRODUCT
UNIT R-1 : Reactor
    REACTIONS:
        H2 + 0.5 O2 -> H2O | conversion=0.5
"""


def run():
    with preview() as address, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome") or shutil.which("chromium"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(address + "/editor")
        expect(page.locator(".palette-item")).to_have_count(33)
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill(SOURCE)
        page.click("#apply-source")
        expect(page.locator("#source-status")).to_have_text(
            "Source synchronized", timeout=15000
        )
        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Named reactions", exact=True
        ).click()
        page.locator("#modal").get_by_role(
            "button", name="＋ Add entry", exact=True
        ).click()
        label_input(page.locator("#modal"), "Reaction name").fill("water_formation")
        label_input(page.locator("#modal"), "Reaction equation").fill(
            "H2 + 0.5 O2 -> H2O"
        )
        label_input(page.locator("#modal"), "Reaction specification").select_option(
            "conversion"
        )
        page.locator(
            "#modal [data-setting=conversion] input:not([type=checkbox])"
        ).fill("0.5")
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()
        page.locator("[data-view=diagram]").click()
        page.locator('[data-unit="R-1"]').press("Enter")
        page.locator("#inspector-tabs").get_by_role(
            "button", name="Reactions", exact=True
        ).click()
        label_input(page.locator("#inspector-content"), "Named reaction").select_option(
            "water_formation"
        )
        expect(
            label_input(page.locator("#inspector-content"), "Reaction equation")
        ).to_be_disabled()
        page.locator("#inspector-content").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        page.locator("[data-view=code]").click()
        assert "water_formation" in page.locator("#code-editor").input_value()

        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Components", exact=True
        ).click()
        water = (
            page.locator("#modal .structured-group")
            .filter(
                has=page.locator(":scope > summary", has_text=re.compile("^Entry$"))
            )
            .nth(2)
        )
        water.get_by_role("button", name="Thermo", exact=True).click()
        water.locator(":scope > .add-field select").select_option("Tc")
        water.locator(":scope > .add-field button").click()
        label_input(water, "Critical temperature (K)").fill("647.096")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Interaction data", exact=True
        ).click()
        parameter_group = (
            page.locator("#modal .structured-group")
            .filter(
                has=page.locator(
                    ":scope > summary", has_text=re.compile("^Interaction parameters")
                )
            )
            .first
        )
        parameter_group.get_by_role("button", name="＋ Add entry", exact=True).click()
        label_input(parameter_group, "Component 1").select_option("H2O")
        label_input(parameter_group, "Component 2").select_option("H2")
        parameter_group.locator("[data-setting=alpha] input[type=checkbox]").check()
        parameter_group.locator("[data-setting=alpha] input:not([type=checkbox])").fill(
            "0.3"
        )
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()

        # Kinetics remain explicit about activation energy, basis and units.
        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Named reactions", exact=True
        ).click()
        label_input(page.locator("#modal"), "Reaction specification").select_option(
            "power_law"
        )
        page.locator("#modal [data-setting=a] input:not([type=checkbox])").fill("0.1")
        expect(page.locator("#modal")).to_contain_text("Activation-energy units")
        expect(page.locator("#modal")).to_contain_text("Reaction-rate units")
        expect(page.locator("#modal")).to_contain_text("Reaction order · H2")
        page.screenshot(path=REPORT / "guided-kinetics.png", full_page=True)
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()

        # Correlation coefficients survive edits and model selection.
        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Components", exact=True
        ).click()
        water = (
            page.locator("#modal .structured-group")
            .filter(
                has=page.locator(":scope > summary", has_text=re.compile("^Entry$"))
            )
            .nth(2)
        )
        water.get_by_role("button", name="Advanced", exact=True).click()
        water.locator(":scope > .add-field select").select_option(
            "property_correlations"
        )
        water.locator(":scope > .add-field button").click()
        water.get_by_label("Property to correlate", exact=True).select_option("Cpl")
        water.get_by_role("button", name="Add correlation", exact=True).click()
        label_input(water, "Correlation equation").select_option("poly_x")
        water.locator("[data-setting=a] input[type=checkbox]").check()
        water.locator("[data-setting=a] input:not([type=checkbox])").fill("75.3")
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()
        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Thermodynamics", exact=True
        ).click()
        label_input(page.locator("#modal"), "Thermo method").select_option("NRTL-BV")
        page.locator("#modal [data-setting=correlation] input[type=checkbox]").check()
        page.locator("#modal [data-setting=correlation] select").select_option("HOC")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Recycle", exact=True
        ).click()
        label_input(page.locator("#modal"), "Recycle method").select_option("DIRECT")
        page.locator("#modal [data-setting=damping] input[type=checkbox]").check()
        page.locator("#modal [data-setting=damping] input:not([type=checkbox])").fill(
            "0.8"
        )
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()
        page.locator("[data-view=code]").click()
        source = page.locator("#code-editor").input_value()
        assert "HOC" in source and "DIRECT" in source and "75.3" in source

        # A component's optional distribution has a paired size/fraction editor.
        page.click("#project-button")
        page.locator("#modal .vertical-tabs").get_by_role(
            "button", name="Components", exact=True
        ).click()
        water = (
            page.locator("#modal .structured-group")
            .filter(
                has=page.locator(":scope > summary", has_text=re.compile("^Entry$"))
            )
            .nth(2)
        )
        water.get_by_role("button", name="Solids", exact=True).click()
        water.locator(":scope > .add-field select").select_option(
            "particle_size_distribution"
        )
        water.locator(":scope > .add-field button").click()
        label_input(water, "Particle distribution model").select_option("discrete")
        water.get_by_role("button", name="Add size class", exact=True).click()
        water.get_by_label("Diameter 1", exact=True).fill("0.00001")
        water.get_by_label("Fraction 1", exact=True).fill("1")
        page.screenshot(path=REPORT / "guided-particles.png", full_page=True)
        # This fluid test component should not acquire a particle override.
        water.get_by_role(
            "button", name="Remove Particle size distribution", exact=True
        ).click()
        page.locator("#modal").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator("#modal")).not_to_be_visible()
        # Reaction controls follow runtime capabilities for every registered unit.
        templates = page.request.get(address + "/api/unit-templates").json()
        source = SOURCE.split("STREAM Feed", 1)[0]
        pump_id = None
        for index, unit_type in enumerate(templates):
            identifier = f"Review-{index}"
            source += f"\nUNIT {identifier} : {unit_type}\n"
            if unit_type == "Pump":
                pump_id = identifier
                source += (
                    "    REACTIONS:\n        H2 + 0.5 O2 -> H2O | conversion=0.5\n"
                )
        page.locator("[data-view=code]").click()
        page.locator("#code-editor").fill(source)
        page.click("#apply-source")
        expect(page.locator("#source-status")).to_have_text(
            "Source synchronized", timeout=15000
        )
        page.locator("[data-view=diagram]").click()
        page.click("#fit-button")
        supported = 0
        for index, (unit_type, template) in enumerate(templates.items()):
            page.locator(f'[data-unit="Review-{index}"]').press("Enter")
            tab = page.locator("#inspector-tabs").get_by_role(
                "button", name="Reactions", exact=True
            )
            if template["supports_reactions"]:
                supported += 1
                expect(tab).to_be_enabled()
                tab.click()
                expect(
                    page.locator("#inspector-content").get_by_role(
                        "button", name="＋ Add entry", exact=True
                    )
                ).to_be_visible()
            else:
                expect(tab).to_have_count(0)
        assert supported == 6 and len(templates) - supported == 27
        # Hiding unsupported inputs must not drop imported data on another edit.
        page.locator(f'[data-unit="{pump_id}"]').press("Enter")
        expect(page.locator("#inspector-heading")).to_contain_text(
            "does not support reactions"
        )
        label_input(page.locator("#inspector-content"), "Id").fill("PreservedPump")
        page.locator("#inspector-content").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(page.locator('[data-unit="PreservedPump"]')).to_have_count(1)
        reactions = page.evaluate("""() => {
            const id = JSON.parse(localStorage.getItem('pfdsim.last.v1'));
            return JSON.parse(localStorage.getItem('pfdsim.laboratories.v1'))[id].pfd.units.find(u => u.id === 'PreservedPump').reactions;
        }""")
        assert len(reactions) == 1, reactions
        assert reactions[0]["equation"] == "H2 + 0.5 O2 -> H2O", reactions
        assert reactions[0]["parameters"]["conversion"] == "0.5", reactions
        label_input(page.locator("#inspector-content"), "Unit type").select_option(
            "Reactor"
        )
        page.locator("#inspector-content").get_by_role(
            "button", name="Apply changes", exact=True
        ).click()
        expect(
            page.locator("#inspector-tabs").get_by_role(
                "button", name="Reactions", exact=True
            )
        ).to_be_enabled()
        assert not errors, errors
        browser.close()
        print(
            "PASS: configuration editors, reaction tabs visible for 6 supported models and hidden for 27 unsupported models, imported-reaction preservation, and unit-type changes"
        )
        print("Artifacts:", REPORT)


if __name__ == "__main__":
    print("Configuration browser report:", REPORT, flush=True)
    run()
