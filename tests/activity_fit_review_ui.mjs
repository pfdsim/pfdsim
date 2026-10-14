// Exercise real review/session modules; keep transport and DOM boundaries local.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { SourceTextModule, SyntheticModule, createContext } from "node:vm";
import { randomUUID } from "node:crypto";

class Node {
  get value() { return this._value; }
  set value(value) { this._value = String(value ?? ""); }
  constructor(tag, attributes = {}, text = "") {
    this.tag = tag;
    Object.assign(this, attributes);
    this.textContent = text;
    this.children = [];
    this.value = "";
    this.listeners = new Map();
  }
  append(...children) { this.children.push(...children); }
  close() { this.closed = true; for (const fn of [...(this.listeners.get("close") || [])]) fn(); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, fn) { const entries = this.listeners.get(type) || new Set(); entries.add(fn); this.listeners.set(type, entries); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  scrollIntoView() { this.scrolled = true; }
  setCustomValidity(message) { this.validityMessage = message; }
  querySelectorAll(selector) {
    assert.equal(selector, "button", `Unexpected test DOM selector ${selector}`);
    return descendants(this).filter(item => item.tag === "button");
  }
}
const nodes = new Map();
const element = (tag, attributes, text) => {
  const created = new Node(tag, attributes, text);
  if (attributes?.id) nodes.set(attributes.id, created);
  return created;
};
const node = id => {
  if (!nodes.has(id)) nodes.set(id, element("div"));
  return nodes.get(id);
};
const descendants = root => [root, ...root.children.flatMap(descendants)];
const content = root => descendants(root).map(item => item.textContent).join("\n");
let dialog;
const calls = [];
const downloads = [];
const apiReplies = new Map();
const apiErrors = new Map();
let publicationJob = { status: "completed", output: { status: "published", id: "fit" } };
const storage = new Map();
let account = { user: null };
let rejectSave = false;
let writable = true;
const context = createContext({
  Node, console, structuredClone, crypto: { randomUUID },
  document: { createTextNode: text => element("text", {}, text) },
});
const exports = {
  "./common.js": {
    $: node, element, svgElement: element,
    guarded: fn => fn.name === "initialize" ? () => {} : fn,
    modal: (_title, body) => { dialog = body; node("modal").closed = false; },
    api: async (path, data) => {
      calls.push({ path, data });
      if (apiErrors.has(path)) throw apiErrors.get(path);
      if (apiReplies.has(path)) return structuredClone(apiReplies.get(path));
      if (path.startsWith("/api/fitting/sessions/")) {
        if (rejectSave) throw Object.assign(new Error("This saved fit changed"), { status: 409 });
        return { version: 1, updated: 12345 };
      }
      if (path === "/api/fitting/sessions") return { sessions: [] };
      return path.endsWith("/publish") || path.endsWith("/withdraw") ? { job_id: "publication" } : { submissions: [] };
    },
    pollJob: async (_id, progress) => {
      if (progress) progress({ progress: ["Building reviewed parameters"], status: "running" });
      return structuredClone(publicationJob);
    },
    formatNumber: value => String(value),
    toast() {}, getSession: async () => account,
    download: (...args) => downloads.push(args),
    readLocal: (key, fallback) => storage.has(key) ? structuredClone(storage.get(key)) : fallback,
    writeLocal: (key, value) => { if (!writable) return false; storage.set(key, structuredClone(value)); return true; },
  },
  "./persistence.js": { initializeLibrary() {}, queueCloud() {} },
  "./fitting-observations.js": { appendObservations() {}, repairObservationIds() {}, mergeSigma() {} },
  "./fitting-plots.js": { renderObjectivePlots() {} },
};
const source = await readFile(new URL("../static/js/parameter-fitting.js", import.meta.url), "utf8");
const sessionSource = await readFile(new URL("../static/js/fitting-sessions.js", import.meta.url), "utf8");
const sessionModule = new SourceTextModule(sessionSource, { context });
const adminModule = new SourceTextModule(await readFile(new URL("../static/js/fitting-admin.js", import.meta.url), "utf8"), { context });
const uiModule = new SourceTextModule(await readFile(new URL("../static/js/fitting-ui.js", import.meta.url), "utf8"), { context });
const realModules = new Map([["./fitting-sessions.js", sessionModule], ["./fitting-admin.js", adminModule], ["./fitting-ui.js", uiModule]]);
const module = new SourceTextModule(source + "\nexport { reconcileFitParameters, reconcileModelChange, renderObservations, removeObservationIds, renderResult }; export function setDisplayedResult(value) { result = value; } export function setParameterModel(value) { parameterModel = value; } export function setLawCatalog(forms) { catalog = { forms, missing_tokens: ['', '?', 'none'], kinds: ['UCST', 'LCST', 'VLLE', 'HE', 'GAMMA_INF'], scales: {}, vapor_requirements: {}, psat_forms: {} }; } export function setObservationRows(rows) { observations = rows; }", { context });
const linked = new Map();
async function link(path) {
  if (linked.has(path)) return linked.get(path);
  if (realModules.has(path)) {
    const dependency = realModules.get(path);
    linked.set(path, dependency);
    await dependency.link(link);
    return dependency;
  }
  const values = exports[path];
  assert.ok(values, `Unexpected import ${path}`);
  const dependency = new SyntheticModule(Object.keys(values), function () {
    for (const [key, value] of Object.entries(values)) this.setExport(key, value);
  }, { context });
  linked.set(path, dependency);
  return dependency;
}
await module.link(link);
await module.evaluate();

for (const psat of [null, { form: "antoine", Tmin_K: 273, Tmax_K: 400, coefficients: { A: 4.296305 } }]) {
  await adminModule.namespace.reviewFit("fit", {
    status: "approved", model: "NRTL", source: { citation: "Reviewed pressure correction" },
    result: {
      success: false, warnings: ["Parameters at bounds: 12.linear"],
      components: ["Fit_1", "Fit_2"], component_names: ["Acetone", "Water"],
      request: { psat: [psat, null], weights: {} }, parameters: {}, objectives: {},
      property_provenance: [{
        property: "Psat", component: "Fit_1", T_K: 289.55, value: 0.19993139808264476,
        source: "Fitting pressure correction", notes: "Supplied correlation; validity 273–400 K",
      }],
    },
    events: [],
  });
  const text = content(dialog);
  assert.match(text, /Fitting saturation-pressure basis/);
  assert.match(text, /do not block publication/);
  assert.match(text, /Acetone/);
  assert.match(text, /289.55/);
  assert.match(text, /0.19993139808264476/);
  assert.match(text, /validity 273–400 K/);
  const details = JSON.parse(descendants(dialog).find(item => item.tag === "pre").textContent);
  assert.deepEqual(details.fitting_psat_definitions, [psat, null]);
  assert.deepEqual(details.warnings, ["Parameters at bounds: 12.linear"]);
  const publish = descendants(dialog).find(item => item.textContent === "Publish to runtime");
  assert.ok(publish && !publish.disabled);
  await publish.onclick();
  assert.equal(calls.at(-2).path, "/api/fitting/admin/publish");
  assert.deepEqual({ ...calls.at(-2).data }, { id: "fit", notes: "" });
}
console.log("Psat review information and publication action passed.");

function reviewSubmission(status = "submitted") {
  return {
    status, version: 4, model: "NRTL", source: { citation: "Reviewed source", doi: "source-doi" },
    review_notes: "Original review notes", events: [{ action: "submitted" }],
    result: {
      components: ["ethanol", "water"], component_names: ["Ethanol", "Water"], method: "NRTL",
      success: false, warnings: ["Requires scientific review"],
      request: { weights: { HE: 1 }, psat: [null, null] },
      parameters: { alpha12: .3, tau12_d: 120 }, objectives: {},
    },
  };
}
const reviewAction = label => descendants(dialog).find(item => item.textContent === label);
const publicationCalls = () => calls.filter(call => ["/api/fitting/admin/publish", "/api/fitting/admin/withdraw"].includes(call.path));
calls.length = 0;
const pending = reviewSubmission();
await adminModule.namespace.reviewFit("pending", pending);
assert.equal(reviewAction("Publish to runtime"), undefined);
node("fit-admin-review-notes").value = "Checked the primary source";
apiReplies.set("/api/fitting/admin/review", { submission: { ...pending, status: "approved", version: 5 } });
await reviewAction("Approve").onclick();
assert.deepEqual({ ...calls.find(call => call.path === "/api/fitting/admin/review").data }, {
  id: "pending", action: "approve", notes: "Checked the primary source", version: 4,
});
assert.equal(publicationCalls().length, 0); // Approval never activates runtime parameters.
assert.match(node("fit-admin-review-status").textContent, /Approved and saved/);
assert.equal(reviewAction("Approve"), undefined);
assert.ok(reviewAction("Publish to runtime"));

calls.length = 0;
await adminModule.namespace.reviewFit("pending", pending);
node("fit-admin-review-notes").value = "Insufficient source evidence";
apiReplies.set("/api/fitting/admin/review", { submission: { ...pending, status: "rejected", version: 5 } });
await reviewAction("Reject").onclick();
assert.deepEqual({ ...calls.find(call => call.path === "/api/fitting/admin/review").data }, {
  id: "pending", action: "reject", notes: "Insufficient source evidence", version: 4,
});
assert.equal(publicationCalls().length, 0);
assert.equal(reviewAction("Publish to runtime"), undefined);

calls.length = 0;
apiReplies.delete("/api/fitting/admin/review");
apiErrors.set("/api/fitting/admin/review", Object.assign(new Error("Review version is stale"), { status: 409 }));
await adminModule.namespace.reviewFit("pending", pending);
node("fit-admin-review-notes").value = "Keep my attempted review";
await reviewAction("Approve").onclick();
assert.match(node("fit-admin-review-status").textContent, /Review version is stale/);
assert.equal(node("fit-admin-review-notes").value, "Keep my attempted review");
assert.ok(descendants(dialog).filter(item => item.tag === "button").every(item => !item.disabled));
assert.equal(publicationCalls().length, 0);
assert.equal(node("modal").closed, false);
apiErrors.clear();

for (const [status, label] of [["approved", "Publish to runtime"], ["publishing", "Recover interrupted publication"], ["published", "Rebuild published fit"]]) {
  calls.length = 0;
  await adminModule.namespace.reviewFit("fit", reviewSubmission(status));
  node("fit-admin-review-notes").value = "Publication audit note";
  if (status !== "approved") {
    assert.equal(reviewAction("Approve"), undefined);
    assert.equal(reviewAction("Reject"), undefined);
  }
  await reviewAction(label).onclick();
  assert.deepEqual({ ...publicationCalls()[0].data }, { id: "fit", notes: "Publication audit note" });
  assert.equal(publicationCalls()[0].path, "/api/fitting/admin/publish");
  assert.match(node("fit-admin-status").textContent, /^published: fit\./);
}
calls.length = 0;
await adminModule.namespace.reviewFit("fit", reviewSubmission("published"));
node("fit-admin-review-notes").value = "";
await assert.rejects(reviewAction("Withdraw and rebuild").onclick(), /Explain the withdrawal/);
assert.equal(publicationCalls().length, 0);
node("fit-admin-review-notes").value = "Wrong source convention";
publicationJob = { status: "completed", output: { status: "revoked", id: "fit" } };
await reviewAction("Withdraw and rebuild").onclick();
assert.deepEqual({ ...publicationCalls()[0].data }, { id: "fit", notes: "Wrong source convention" });
assert.equal(publicationCalls()[0].path, "/api/fitting/admin/withdraw");
assert.match(node("fit-admin-status").textContent, /^revoked: fit\./);
publicationJob = { status: "failed", error: "Publication validation rejected the coefficients" };
await assert.rejects(adminModule.namespace.adminPublication({ id: "fit" }), /Publication validation rejected/);
assert.doesNotMatch(node("fit-admin-status").textContent, /^published:/);
publicationJob = { status: "completed", output: { status: "published", id: "fit" } };
apiErrors.set("/api/fitting/admin/publish", new Error("Could not queue publication"));
node("fit-admin-status").textContent = "Not published";
await assert.rejects(adminModule.namespace.adminPublication({ id: "fit" }), /Could not queue publication/);
assert.equal(node("fit-admin-status").textContent, "Not published");
apiErrors.clear();

const specialId = "review /?&id";
const detailPath = `/api/fitting/admin/submissions/${encodeURIComponent(specialId)}`;
apiReplies.set(detailPath, { submission: reviewSubmission("approved") });
await adminModule.namespace.reviewFit(specialId);
assert.equal(calls.at(-1).path, detailPath);
assert.equal(reviewAction("View fit assessment").href, `/parameter-fitting?review=${encodeURIComponent(specialId)}`);
apiReplies.clear();
console.log("Review/approval/rejection, optimistic versions, publication/recovery/withdrawal and failure propagation passed.");

for (const model of ["NRTL", "UNIQUAC"]) {
  const manual = reviewSubmission("approved");
  manual.model = manual.result.method = model;
  Object.assign(manual.result, {
    submission_origin: "manual_parameters",
    reported_fit: { method: "Source weighted least squares", statistics: "AAD 0.5%\n42 source observations" },
    manual_input: { model, basis: "tau", values: { "12": .6, "21": 1.2 } },
  });
  await adminModule.namespace.reviewFit("manual", manual);
  assert.match(content(dialog), /PFDSim did not run a regression/);
  assert.match(content(dialog), /Source-reported fitting method: Source weighted least squares/);
  assert.match(content(dialog), /AAD 0.5%\n42 source observations/);
  assert.ok(descendants(dialog).some(item => item.tag === "pre" && item.textContent === JSON.stringify(manual.result.manual_input, null, 2)));
  assert.equal(reviewAction("View fit assessment"), undefined);
  downloads.length = 0;
  reviewAction("Download full provenance").onclick();
  assert.deepEqual(JSON.parse(downloads[0][0]), manual);
  reviewAction("Download fit result").onclick();
  assert.deepEqual(JSON.parse(downloads[1][0]), manual.result);
  apiReplies.set("/api/fitting/export", { pfd_text: "Sourced manual parameters PFD" });
  await reviewAction("Download fitted PFD").onclick();
  assert.equal(downloads[2][0], "Sourced manual parameters PFD");
  assert.deepEqual(calls.at(-1).data.result, manual.result);
  assert.ok(reviewAction("Publish to runtime"));
}
apiReplies.clear();
console.log("Manual NRTL/UNIQUAC entries preserve source-reported method/statistics, raw input and provenance exports.");

const fitted = {
  components: ["A", "B"], component_names: ["Acetone", "Water"], method: "UNIQUAC-PR", model: "UNIQUAC",
  request: {
    components: ["original-alias-A", "original-alias-B"], model: "UNIQUAC", vapor: "PR", form: "constant_inverse_anchored",
    alpha: .3, fit_alpha: false, T_ref_K: 310, starts: 4, max_nfev: 700, seed: 42,
    cv: { method: "none", folds: 5 }, extrapolation: "unrestricted", online_lookup: false,
    estimate_properties: false, allow_hoc_eta_default: false,
    observations: [{ id: "paper-1", kind: "VLE", T_K: 310, P_bar: 1, x1: .5, y1: .7, pin: true, sigma: { log_fugacity: .02 } }],
    weights: { VLE: 2, HE: 3 }, scales: { log_fugacity: .02, HE_J_mol: 20 },
    initial: { "12.constant": 0 }, bounds: { "12.constant": [-2, 2] },
    vapor_parameters: [{ model: "PR", field: "kij", value: .03, fit: true }],
    psat: [{ form: "antoine", coefficients: { A: 4.2, B: 1230, C: -43 }, Tmin_K: 273, Tmax_K: 400, temperature_unit: "K", pressure_unit: "bar" }, null],
    component_properties: [{ Tc_K: 510 }, {}], import_report: [],
  },
  coefficients: { "12.constant": .5, "vapor.PR.kij": .04 },
  rq: [{ r: 2.5, q: 2 }, { r: .92, q: 1.4 }],
  pfd_text: "COMPONENTS:\n    A | Acetone\n    B | Water\n", pfd: { components: [{ symbol: "A" }, { symbol: "B" }], metadata: { thermo_method: "UNIQUAC-PR" } },
  points: [], parameters: {}, objectives: {}, optimizer: {}, warnings: [], cross_validation: { folds: [] },
};
const sourceCitation = { citation: "Original source", doi: "original-doi", notes: "Original notes" };
module.namespace.setDisplayedResult({ ...fitted, success: false, warnings: ["Fit needs review"] });
module.namespace.renderResult();
assert.equal(node("fit-submit").disabled, false);
assert.equal(node("fit-admin-direct").disabled, false);
assert.match(content(node("fit-warnings")), /Fit needs review/);
module.namespace.setDisplayedResult(null);
const document = sessionModule.namespace.fittingSessionFromResult(fitted, sourceCitation);
assert.equal(document.state.controls.comp1, "A");
assert.equal(document.state.controls.model, "UNIQUAC");
assert.equal(document.state.controls.vapor, "PR");
assert.equal(document.state.controls.law, "constant_inverse_anchored");
assert.equal(document.state.controls.r1, "2.5");
assert.equal(document.state.controls.tref, "310");
assert.equal(document.state.controls["source-doi"], "original-doi");
assert.equal(JSON.parse(document.state.controls.initial)["12.constant"], .5);
assert.deepEqual(document.state.observations, fitted.request.observations);
assert.deepEqual(document.state.psatValues[0], { ...fitted.request.component_properties[0], ...fitted.request.psat[0], ...fitted.request.psat[0].coefficients });
assert.deepEqual(document.state.vaporParameters, fitted.request.vapor_parameters);
assert.equal(document.state.definitionProject.text, fitted.pfd_text);
document.state.controls.alpha = "changed";
assert.equal(fitted.request.alpha, .3); // Session conversion never mutates the report.

function setup(dirty = true) {
  storage.clear(); calls.length = 0; account = { user: null }; rejectSave = false; writable = true;
  let state = { controls: { input: "", model: "NRTL" }, observations: dirty ? [{ id: "current", kind: "HE", T_K: 300, x1: .5, HE_J_mol: 99 }] : [] };
  let active = false;
  const library = sessionModule.namespace.fittingSessionLibrary({
    capture: () => structuredClone(state), restore: saved => { state = structuredClone(saved); }, canOpen: () => !active,
  });
  return { library, state: () => state, active: value => { active = value; } };
}
const reviewed = { status: "published", model: "UNIQUAC", source: sourceCitation, result: fitted, events: [] };
async function view() {
  await adminModule.namespace.reviewFit("fit", reviewed);
  const assessment = descendants(dialog).find(item => item.textContent === "View fit assessment");
  assert.equal(assessment.tag, "a");
  assert.equal(assessment.href, "/parameter-fitting?review=fit");
  assert.equal(assessment.onclick, undefined); // Navigation is handled by the fitter's startup.
  // The real browser check covers that startup wiring. Exercise the same public
  // session operation here for all unsaved-work and account/storage failures.
  return page.library.openReviewedFit(sessionModule.namespace.fittingSessionFromResult(fitted, sourceCitation));
}
const button = label => descendants(dialog).find(item => item.textContent === label);
let page = setup();
let opening = view(); await new Promise(resolve => setImmediate(resolve));
assert.match(content(dialog), /unsaved fitting work/);
assert.equal(page.state().observations[0].id, "current");
button("Cancel").onclick(); await opening;
assert.equal(page.state().observations[0].id, "current");

opening = view(); await new Promise(resolve => setImmediate(resolve));
button("Open without saving").onclick(); await opening;
assert.deepEqual(page.state().observations, fitted.request.observations);
assert.equal(page.state().controls.model, "UNIQUAC");
assert.equal(page.state().controls.vapor, "PR");
assert.equal(page.state().result.method, "UNIQUAC-PR");
assert.equal(storage.get("pfdsim.fit-session-previous.v1").state.observations[0].id, "current");
// An unchanged loaded fit needs no repeated save prompt.
await view(); assert.deepEqual(page.state().observations, fitted.request.observations);

page = setup(); opening = view(); await new Promise(resolve => setImmediate(resolve));
node("modal").close(); await opening;
assert.equal(page.state().observations[0].id, "current");

page = setup(); account = { user: { id: "root" } };
opening = view(); await new Promise(resolve => setImmediate(resolve));
node("fit-switch-name").value = "Saved before reviewing";
await button("Save and open fit").onclick(); await opening;
const saved = calls.find(call => call.path.startsWith("/api/fitting/sessions/"));
assert.equal(saved.data.document.name, "Saved before reviewing");
assert.equal(saved.data.document.state.observations[0].id, "current");
assert.deepEqual(page.state().observations, fitted.request.observations);

page = setup(); account = { user: { id: "root" } }; rejectSave = true;
opening = view(); await new Promise(resolve => setImmediate(resolve));
await button("Save and open fit").onclick();
assert.equal(page.state().observations[0].id, "current");
assert.match(content(dialog), /This saved fit changed/);
button("Cancel").onclick(); await opening;

page = setup(); writable = false;
opening = view(); await new Promise(resolve => setImmediate(resolve));
button("Open without saving").onclick();
await assert.rejects(opening, /Could not preserve the current draft/);
assert.equal(page.state().observations[0].id, "current");

page = setup(false); await view();
assert.deepEqual(page.state().observations, fitted.request.observations);
page.active(true);
await assert.rejects(view(), /Wait for the active calculation/);

// The shared save path still supports the existing Saved fits dialog.
page = setup();
await page.library.open();
node("fit-session-name").value = "Named original draft";
await node("fit-session-save").onclick();
let records = Object.values(storage.get("pfdsim.fit-sessions.v1"));
assert.equal(records.length, 1);
assert.equal(records[0].name, "Named original draft");
node("fit-session-name").value = "Separate copy";
await node("fit-session-copy").onclick();
records = Object.values(storage.get("pfdsim.fit-sessions.v1"));
assert.equal(records.length, 2);
assert.deepEqual(records.map(record => record.name).sort(), ["Named original draft", "Separate copy"]);
node("fit-session-name").value = "Updated copy";
await node("fit-session-save").onclick();
assert.equal(Object.keys(storage.get("pfdsim.fit-sessions.v1")).length, 2);
console.log("Reviewed-fit restoration, save/cancel choices, provenance, active-job and storage protections passed.");

module.namespace.setLawCatalog({
  constant: ["constant"], inverse: ["inverse"], constant_inverse: ["constant", "inverse"],
  constant_inverse_anchored: ["constant", "inverse", "anchored"],
  constant_inverse_linear: ["constant", "inverse", "linear"],
  constant_inverse_anchored_linear: ["constant", "inverse", "anchored", "linear"],
  full: ["constant", "inverse", "anchored", "linear", "quadratic"],
});
node("fit-law").value = "constant_inverse_linear";
node("fit-model").value = "NRTL";
node("fit-free-alpha").checked = true;
node("fit-vapor").value = "PR";
for (const kind of ["UCST", "VLLE"]) node(`fit-weight-${kind}`).value = "1";
module.namespace.setObservationRows([
  { id: "paper", kind: "UCST", T_K: 300 },
  { id: "paper", kind: "VLLE", T_K: 300, P_bar: 1 },
]);
const vaporGroup = {
  dataset: { model: "PR", field: "kij", scale: "1" },
  querySelector: selector => ({ checked: true, value: selector.includes('"lower"') ? -.5 : selector.includes('"upper"') ? .5 : .02 }),
};
node("fit-vapor-parameters").children = [vaporGroup];
const starts = { "12.constant": .4, "12.inverse": 450, "12.anchored": -5, "21.constant": .7, "21.inverse": -11, "21.anchored": .6, alpha12: .55, "vapor.PR.kij": .02, "critical_x1.paper": .3, "12.typo": 2, "12.anchored.": 3 };
node("fit-initial").value = JSON.stringify(starts);
node("fit-bounds").value = JSON.stringify({ "12.anchored": [-6, 6], "21.inverse": [-30, 30], "vlle_xa.paper": [.01, .8] });
module.namespace.setParameterModel("NRTL");
node("fit-model").value = "UNIQUAC";
module.namespace.reconcileModelChange();
assert.deepEqual(JSON.parse(node("fit-initial").value), { "vapor.PR.kij": .02, "critical_x1.paper": .3 });
assert.deepEqual(JSON.parse(node("fit-bounds").value), { "vlle_xa.paper": [.01, .8] });
node("fit-model").value = "NRTL";
module.namespace.setParameterModel("NRTL");
node("fit-initial").value = JSON.stringify(starts);
node("fit-bounds").value = JSON.stringify({ "12.anchored": [-6, 6], "21.inverse": [-30, 30], "vlle_xa.paper": [.01, .8] });
module.namespace.reconcileFitParameters();
const expected = { ...starts }; delete expected["12.anchored"]; delete expected["21.anchored"];
assert.deepEqual(JSON.parse(node("fit-initial").value), expected);
assert.deepEqual(JSON.parse(node("fit-bounds").value), { "21.inverse": [-30, 30], "vlle_xa.paper": [.01, .8] });
assert.ok(!("12.linear" in JSON.parse(node("fit-initial").value))); // Backend supplies zero; anchored C is not a D slope.
node("fit-law").value = "full";
module.namespace.reconcileFitParameters();
assert.deepEqual(JSON.parse(node("fit-initial").value), expected);
const abcd = { ...expected, "12.anchored": -5, "12.linear": .002 };
node("fit-initial").value = JSON.stringify({ ...abcd, "12.quadratic": .000001, "21.quadratic": -.000001 });
node("fit-bounds").value = JSON.stringify({ "12.linear": [-.01,.01], "12.quadratic": [-.001,.001] });
node("fit-law").value = "constant_inverse_anchored_linear";
module.namespace.reconcileFitParameters();
assert.deepEqual(JSON.parse(node("fit-initial").value), abcd);
assert.deepEqual(JSON.parse(node("fit-bounds").value), { "12.linear": [-.01,.01] });
node("fit-initial").value = JSON.stringify(expected);
node("fit-bounds").value = JSON.stringify({ "21.inverse": [-30,30], "vlle_xa.paper": [.01,.8] });
node("fit-law").value = "constant";
module.namespace.reconcileFitParameters();
delete expected["12.inverse"]; delete expected["21.inverse"];
assert.deepEqual(JSON.parse(node("fit-initial").value), expected);
assert.deepEqual(JSON.parse(node("fit-bounds").value), { "vlle_xa.paper": [.01, .8] });
node("fit-initial").value = '{"12.constant":';
node("fit-bounds").value = '[]';
module.namespace.reconcileFitParameters();
assert.equal(node("fit-initial").value, '{"12.constant":');
assert.equal(node("fit-bounds").value, '[]');
console.log("Temperature-law changes preserve compatible seeds, bounds and auxiliary parameters; obsolete terms are removed without reinterpreting units.");

function parameterDraft() {
  node("fit-law").value = "constant_inverse";
  node("fit-model").value = "NRTL";
  node("fit-free-alpha").checked = true;
  node("fit-vapor").value = "PR";
  node("fit-vapor-parameters").children = [vaporGroup];
  module.namespace.setObservationRows([
    { id: "critical.with.dots", kind: "UCST", T_K: 300 },
    { id: "invariant", kind: "VLLE", T_K: 300, P_bar: 1 },
  ]);
  const values = { "12.constant": 1, "21.inverse": 200, alpha12: .4, "vapor.PR.kij": .02,
    "critical_x1.critical.with.dots": .3, "vlle_xa.invariant": .1, "vlle_gap.invariant": .8, typo: 9 };
  node("fit-initial").value = JSON.stringify(values);
  node("fit-bounds").value = JSON.stringify(Object.fromEntries(Object.keys(values).map(key => [key, [-1, 1]])));
  return values;
}
for (const change of ["UNIQUAC", "fixed-alpha", "IDEAL", "fixed-vapor", "validation", "known-compositions", "disabled-data", "kind-change", "removed-rows"]) {
  const expected = parameterDraft();
  if (change === "UNIQUAC") { node("fit-model").value = "UNIQUAC"; delete expected.alpha12; }
  if (change === "fixed-alpha") { node("fit-free-alpha").checked = false; delete expected.alpha12; }
  if (change === "IDEAL") { node("fit-vapor").value = "IDEAL"; delete expected["vapor.PR.kij"]; }
  if (change === "fixed-vapor") {
    node("fit-vapor-parameters").children = [{ ...vaporGroup, querySelector: selector => ({ checked: !selector.includes('"fit"'), value: .02 }) }];
    delete expected["vapor.PR.kij"];
  }
  if (["validation", "known-compositions", "disabled-data", "kind-change", "removed-rows"].includes(change)) {
    module.namespace.setObservationRows(change === "removed-rows" ? [] : [
      { id: "critical.with.dots", kind: change === "kind-change" ? "HE" : "UCST", T_K: 300,
        ...(change === "validation" ? { validation_only: true } : change === "known-compositions" ? { x1: .4 } : change === "disabled-data" ? { weight: 0 } : {}) },
      { id: "invariant", kind: change === "kind-change" ? "GAMMA_INF" : "VLLE", T_K: 300, P_bar: 1,
        ...(change === "validation" ? { validation_only: true } : change === "known-compositions" ? { x1_alpha: .1, x1_beta: .9 } : change === "disabled-data" ? { weight: 0 } : {}) },
    ]);
    delete expected["critical_x1.critical.with.dots"];
    delete expected["vlle_xa.invariant"];
    delete expected["vlle_gap.invariant"];
  }
  module.namespace.reconcileFitParameters();
  assert.deepEqual(JSON.parse(node("fit-initial").value), expected, change);
  assert.deepEqual(Object.keys(JSON.parse(node("fit-bounds").value)).sort(), Object.keys(expected).sort(), change);
}
parameterDraft();
node("fit-weight-UCST").value = "0";
module.namespace.reconcileFitParameters();
assert.ok(!("critical_x1.critical.with.dots" in JSON.parse(node("fit-initial").value)));
node("fit-weight-UCST").value = "1";
console.log("Model, alpha, vapor and observation changes remove obsolete optimizer parameters while preserving compatible entries and unknown-key diagnostics.");

// Exercise the real observation controls, including their reconciliation hooks.
parameterDraft();
module.namespace.renderObservations();
let control = descendants(node("fit-table")).find(item => item["aria-label"] === "Validation-only observation critical.with.dots");
control.checked = true; control.onchange();
assert.ok(!("critical_x1.critical.with.dots" in JSON.parse(node("fit-initial").value)));
assert.ok(!("critical_x1.critical.with.dots" in JSON.parse(node("fit-bounds").value)));
control = descendants(node("fit-table")).find(item => item["aria-label"] === "Observation invariant x1_alpha");
control.value = "0.1"; control.oninput();
assert.ok(!("vlle_xa.invariant" in JSON.parse(node("fit-initial").value)));
assert.ok(!("vlle_gap.invariant" in JSON.parse(node("fit-bounds").value)));
assert.equal(JSON.parse(node("fit-initial").value).alpha12, .4);
console.log("Actual observation validation and composition controls reconcile initial values and bounds.");
