import { $, api, element, svgElement, guarded, toast, modal, pollJob, getSession, download, readLocal, writeLocal, formatNumber } from "./common.js";
import { initializeLibrary, queueCloud } from "./persistence.js";
import { appendObservations, repairObservationIds, mergeSigma } from "./fitting-observations.js";
import { renderObjectivePlots } from "./fitting-plots.js";
import { fittingSessionLibrary, fittingSessionFromResult } from "./fitting-sessions.js";
import { table, labeled } from "./fitting-ui.js";
import { adminPublication } from "./fitting-admin.js";

const draftKey = "pfdsim.fitting.v1";
let catalog, observations = [], inputDirty = true, result = null, jobId = null, activeJob = null;
let definitionProject = null, exportProject = null;
let parameterModel = null;
let storageWarningShown = false;
let observationSets = [], importReports = [], manualValues = {};
let psatValues = [{}, {}];
let pendingImport = null;
let sessionLibrary;
const fields = ["comp1", "comp2", "model", "vapor", "law", "alpha", "free-alpha", "r1", "q1", "r2", "q2", "cv", "folds", "tref", "starts", "evaluations", "seed", "extrapolation", "online", "scales", "initial", "bounds", "input", "source", "source-url", "source-doi", "source-notes", "input-method", "data-kind", "estimate-properties", "hoc-eta-default"];
const labels = { VLE: "Vapor–liquid equilibrium", LLE: "Liquid–liquid equilibrium", HE: "Excess enthalpy", GAMMA_INF: "Infinite-dilution γ", AZEOTROPE: "Azeotrope", VLLE: "VLLE / heteroazeotrope", UCST: "Upper critical solution point", LCST: "Lower critical solution point" };
const laws = { constant: "A", inverse: "B/T", constant_inverse: "A + B/T", constant_inverse_anchored: "A + B/T + C h(T)", constant_inverse_linear: "A + B/T + D T", constant_inverse_anchored_linear: "A + B/T + C h(T) + D T", full: "A + B/T + C h(T) + D T + E T²" };
const vaporParameterDefinitions = {
  VDM: [["VDM", "delta_H_residual_J_per_mol", "Cross-association ΔH / J mol⁻¹", 1000], ["VDM", "delta_S_residual_J_per_mol_K", "Cross-association ΔS / J mol⁻¹ K⁻¹", 10]],
  HOC: [["HOC", "eta", "HOC cross-association η", 1]],
  TSONOPOULOS: [["TSONOPOULOS", "kij", "Tsonopoulos kᵢⱼ", 1]],
  PR: [["PR", "kij", "PR kᵢⱼ", 1]],
};
const enthalpyUnits = [["J/mol", "J/mol"], ["kJ/mol", "kJ/mol"], ["cal/mol", "cal/mol"], ["kcal/mol", "kcal/mol"]];
const exampleRows = {
  VLE: { kind: "VLE", T_K: 350, P_bar: 1, x1: 0.3, y1: 0.6 },
  LLE: { kind: "LLE", T_K: 300, x1_alpha: 0.02, x1_beta: 0.85 },
  HE: { kind: "HE", T_K: 300, x1: 0.5, HE_J_mol: 500 },
  GAMMA_INF: { kind: "GAMMA_INF", T_K: 300, gamma1_inf: 3, gamma2_inf: 2 },
  AZEOTROPE: { kind: "AZEOTROPE", T_K: 350, P_bar: 1, x1: 0.7, weight: 5 },
  VLLE: { kind:"VLLE",T_K:350,P_bar:1,x1_alpha:0.02,x1_beta:0.95,y1:0.5,weight:5 },
  UCST: { kind: "UCST", T_K: 330, x1: 0.5 },
  LCST: { kind: "LCST", T_K: 310 },
};
function captureFitState() {
  const controls = Object.fromEntries(fields.map(name => {
    const control = $(`fit-${name}`);
    return [name, control.type === "checkbox" ? control.checked : control.value];
  }));
  const weights = Object.fromEntries(catalog.kinds.map(kind => [kind, $(`fit-weight-${kind}`).value]));
  const sigmaValues = Object.fromEntries(Object.keys(catalog.scales).map(key => [key, $(`fit-sigma-${key}`).value]));
  const exportMapping=result?Object.fromEntries(result.components.map((name,index)=>[name,$(`fit-map-${index}`)?.value])):{};
  return structuredClone({ controls, weights, observations, inputDirty, result, jobId, activeJob, definitionProject, exportProject, vaporParameters: readVaporParameters(), observationSets, importReports, manualValues, sigmaValues, psatValues,pendingImport,
                         selectedScope:$("fit-scope").value,exportScope:$("fit-export-scope").value,exportMapping });
}
function saveDraft() {
  const saved = writeLocal(draftKey,captureFitState());
  if (!saved && !storageWarningShown) { storageWarningShown = true; toast("Browser storage could not save this fitting draft. Download the fit report before leaving this page.", true); }
  if (saved) storageWarningShown = false;
}
function reconcileFitParameters() {
  const active = new Set(catalog.forms[$("fit-law").value]);
  const known = new Set(Object.values(catalog.forms).flat());
  const knownVaporNames = new Set(Object.values(vaporParameterDefinitions).flat().map(([model, field]) => `vapor.${model}.${field}`));
  const vaporNames = new Set(readVaporParameters()
    .filter(spec => spec.fit && spec.model === $("fit-vapor").value)
    .map(spec => `vapor.${spec.model}.${spec.field}`));
  const training = observations.filter(row => !row.validation_only && (row.pin || (row.weight ?? 1) * Number($(`fit-weight-${row.kind}`).value) > 0));
  const latentNames = new Set();
  for (const row of training) {
    if (["UCST", "LCST"].includes(row.kind) && missingValue(row.x1)) latentNames.add(`critical_x1.${row.id}`);
    if (row.kind === "VLLE" && missingValue(row.x1_alpha) && missingValue(row.x1_beta)) {
      latentNames.add(`vlle_xa.${row.id}`); latentNames.add(`vlle_gap.${row.id}`);
    }
    if (row.kind === "LLE" && missingValue(row.x1_alpha) !== missingValue(row.x1_beta)) latentNames.add(`lle_gap.${row.id}`);
  }
  const removed = [];
  for (const field of ["initial", "bounds"]) {
    const control = $(`fit-${field}`);
    let values;
    try { values = JSON.parse(control.value); }
    catch { continue; } // Keep unfinished JSON editable.
    if (!values || Array.isArray(values) || typeof values !== "object") continue;
    const obsolete = Object.keys(values).filter(key => {
      const parts = key.split(".");
      if (parts.length === 2 && ["12", "21"].includes(parts[0]) && known.has(parts[1])) return !active.has(parts[1]);
      if (key === "alpha12") return $("fit-model").value !== "NRTL" || !$("fit-free-alpha").checked;
      if (knownVaporNames.has(key)) return !vaporNames.has(key);
      if (parts.length >= 2 && ["critical_x1", "vlle_xa", "vlle_gap", "lle_gap"].includes(parts[0])) return !latentNames.has(key);
      return false; // Keep unknown keys for the backend to diagnose, and unfinished JSON editable.
    });
    if (!obsolete.length) continue;
    for (const key of obsolete) delete values[key];
    control.value = JSON.stringify(values, null, 2);
    removed.push(`${field}: ${obsolete.join(", ")}`);
  }
  if (removed.length) toast(`Removed parameters no longer fitted by the current settings or observations (${removed.join("; ")}). Compatible values were kept; unspecified terms use default starting values and bounds.`);
  return removed.length > 0;
}
function reconcileModelChange() {
  const selected = $("fit-model").value;
  if (parameterModel === null || parameterModel === selected) {
    parameterModel = selected;
    return false;
  }
  const removed = [];
  for (const field of ["initial", "bounds"]) {
    const control = $(`fit-${field}`);
    let values;
    try { values = JSON.parse(control.value); }
    catch { continue; }
    if (!values || Array.isArray(values) || typeof values !== "object") continue;
    const incompatible = Object.keys(values).filter(key => /^(?:12|21)\./.test(key) || key === "alpha12");
    if (!incompatible.length) continue;
    for (const key of incompatible) delete values[key];
    control.value = JSON.stringify(values, null, 2);
    removed.push(`${field}: ${incompatible.join(", ")}`);
  }
  parameterModel = selected;
  if (removed.length) toast(`Removed ${removed.join("; ")} because NRTL τ coefficients and UNIQUAC ln τ coefficients are not interchangeable.`);
  return removed.length > 0;
}
function updateModel() {
  $("fit-alpha-controls").hidden = $("fit-model").value !== "NRTL";
  $("fit-rq-controls").hidden = $("fit-model").value !== "UNIQUAC";
  $("fit-alpha-controls").querySelectorAll("input").forEach(input => input.disabled = $("fit-model").value !== "NRTL");
  $("fit-rq-controls").querySelectorAll("input").forEach(input => input.disabled = $("fit-model").value !== "UNIQUAC");
  const parameter = $("fit-model").value === "NRTL" ? "τᵢⱼ" : "ln τᵢⱼ";
  $("fit-equation").textContent = `${parameter} = ${laws[$("fit-law").value]}; h(T) = (Tref−T)/T + ln(T/Tref). Each direction has its own coefficients.`;
}
function jsonControl(name) {
  const text = $(`fit-${name}`).value.trim();
  try { return text ? JSON.parse(text) : {}; }
  catch { throw new Error(`${name}: enter valid JSON.`); }
}
function controlsRequest({ includePsat = true } = {}) {
  const request = {
    components: [$("fit-comp1").value.trim(), $("fit-comp2").value.trim()],
    model: $("fit-model").value, vapor: $("fit-vapor").value, form: $("fit-law").value,
    alpha: $("fit-model").value === "NRTL" ? Number($("fit-alpha").value) : 0.3, fit_alpha: $("fit-model").value === "NRTL" && $("fit-free-alpha").checked,
    T_ref_K: Number($("fit-tref").value), starts: Number($("fit-starts").value), max_nfev: Number($("fit-evaluations").value), seed: Number($("fit-seed").value),
    cv: { method: $("fit-cv").value, folds: Number($("fit-folds").value) },
    weights: Object.fromEntries(catalog.kinds.map(kind => [kind, Number($(`fit-weight-${kind}`).value)])),
    scales: jsonControl("scales"), initial: jsonControl("initial"), bounds: jsonControl("bounds"),
    online_lookup: $("fit-online").checked, extrapolation: $("fit-extrapolation").value,
    estimate_properties: $("fit-estimate-properties").checked, allow_hoc_eta_default: $("fit-hoc-eta-default").checked,
    vapor_parameters: readVaporParameters(),
    import_report: importReports,
    ...readPsat(includePsat),
  };
  if (definitionProject) { request.pfd_text = definitionProject.text; request.scope = $("fit-scope").value; }
  if (request.model === "UNIQUAC") {
    const values = ["r1", "q1", "r2", "q2"].map(name => $(`fit-${name}`).value.trim());
    if (values.some(Boolean)) {
      if (!values.every(Boolean)) throw new Error("Enter all four R/Q values, or leave all four blank.");
      request.rq = [{ r: Number(values[0]), q: Number(values[1]) }, { r: Number(values[2]), q: Number(values[3]) }];
    }
  }
  return request;
}
function renderVaporParameters(saved = []) {
  const definitions = vaporParameterDefinitions[$("fit-vapor").value] || [];
  const container = $("fit-vapor-parameters"); container.replaceChildren();
  if (!definitions.length) container.append(element("p", { class: "field-help" }, "This vapor treatment has no editable binary correction here."));
  for (const [model, field, label, scale] of definitions) {
    const previous = saved.find(spec => spec.model === model && spec.field === field);
    const group = element("div", { class: "structured-group", "data-model": model, "data-field": field, "data-scale": scale });
    const enabled = element("input", { type: "checkbox", "data-vapor": "enabled" });
    enabled.checked = !!previous;
    const enabledLabel = element("label"); enabledLabel.append(enabled, document.createTextNode(` Override ${label}`));
    group.append(enabledLabel);
    for (const [key, defaultValue] of [["value", 0], ["lower", -10*scale], ["upper", 10*scale]]) {
      const input = element("input", { type: "number", step: "any", "data-vapor": key }); input.value = previous?.[key] ?? defaultValue;
      const wrapper = element("label", { class: "field" }); wrapper.append(element("span", { class: "field-label" }, key), input); group.append(wrapper);
    }
    const fit = element("input", { type: "checkbox", "data-vapor": "fit" }); fit.checked = previous?.fit ?? false;
    const fitLabel = element("label"); fitLabel.append(fit, document.createTextNode(" Fit this vapor parameter jointly")); group.append(fitLabel);
    group.onchange = () => { reconcileFitParameters(); saveDraft(); }; container.append(group);
  }
}
function readVaporParameters() {
  return Array.from($("fit-vapor-parameters").children).filter(group => group.querySelector('[data-vapor="enabled"]')?.checked).map(group => ({
    model: group.dataset.model, field: group.dataset.field, scale: Number(group.dataset.scale),
    fit: group.querySelector('[data-vapor="fit"]').checked,
    ...Object.fromEntries(["value", "lower", "upper"].map(key => [key, Number(group.querySelector(`[data-vapor="${key}"]`).value)])),
  }));
}
function missingValue(value) {
  return catalog.missing_tokens.includes(String(value ?? "").trim().toLowerCase().replace(/\s+/g, " "));
}
function measurementInput(value, attributes, changed) {
  const node = element("input", { type: "text", inputmode: "decimal", placeholder: "Blank / ?", ...attributes });
  node.value = value ?? "";
  function validate() {
    const text = node.value.trim(), missing = missingValue(text);
    const valid = missing || /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text) && Number.isFinite(Number(text));
    node.setCustomValidity(valid ? "" : "Enter a finite number or a missing indicator such as ?, None, N/A, or a blank.");
    return missing ? undefined : valid ? Number(text) : node.value;
  }
  validate();
  node.oninput = () => changed(validate(), node.value);
  return node;
}
function renderObservations() {
  const counts = {};
  observations.forEach(row => counts[row.kind] = (counts[row.kind] || 0)+1);
  $("fit-data-summary").textContent = `${observations.length} observations · ${Object.entries(counts).map(([kind, count]) => `${kind}: ${count}`).join(" · ")}`;
  const editable = observations.map(row => {
    const select = element("select", { "aria-label": `Observation ${row.id} kind` });
    for (const kind of catalog.kinds) select.append(element("option", { value: kind }, kind)); select.value = row.kind;
    select.onchange = () => {
      const keys = Object.keys(exampleRows[select.value]);
      for (const key of ["P_bar", "x1", "y1", "x1_alpha", "x1_beta", "HE_J_mol", "gamma1_inf", "gamma2_inf"]) if (!keys.includes(key)) delete row[key];
      row.kind = select.value; reconcileFitParameters(); renderObservations(); saveDraft();
    };
    function input(key, type = "number") {
      const attributes = { "aria-label": `Observation ${row.id} ${key}` };
      if (type === "number") return measurementInput(row[key], attributes, value => {
        if (value === undefined) delete row[key]; else row[key] = value;
        reconcileFitParameters();
        if (key === "weight") renderPsat(); saveDraft();
      });
      const node = element("input", { type, ...attributes }); node.value = row[key] ?? "";
      node.oninput = () => { row[key] = node.value; saveDraft(); }; return node;
    }
    const coordinateInput = element("div", { class: "fit-coordinate" });
    const coordinateKeys = {
      VLE: ["P_bar", "x1", "y1"], LLE: ["x1_alpha", "x1_beta"], HE: ["x1", "HE_J_mol"],
      VLLE:["P_bar","x1_alpha","x1_beta","y1"],
      GAMMA_INF: ["gamma1_inf", "gamma2_inf"], AZEOTROPE: ["P_bar", "x1"], UCST: ["x1"], LCST: ["x1"],
    }[row.kind];
    for (const key of coordinateKeys) {
      const label = element("label", { class: "field" }); label.append(element("span", { class: "field-label" }, key), input(key)); coordinateInput.append(label);
    }
    const pin = element("input", { type: "checkbox", "aria-label": `Pin observation ${row.id}` }); pin.checked = !!row.pin;
    pin.disabled = !!row.validation_only;
    pin.onchange = () => { row.pin = pin.checked; reconcileFitParameters(); renderPsat();saveDraft(); };
    const validation = element("input", { type:"checkbox", "aria-label":`Validation-only observation ${row.id}` }); validation.checked = !!row.validation_only;
    validation.onchange = () => { row.validation_only=validation.checked; if(row.validation_only)row.pin=false; reconcileFitParameters(); renderObservations();saveDraft(); };
    const sigma = element("input", { type: "text", "aria-label": `Observation ${row.id} sigma` }); sigma.value = JSON.stringify(row.sigma ?? {});
    sigma.oninput = () => { try { if (missingValue(sigma.value)) delete row.sigma; else row.sigma = JSON.parse(sigma.value); sigma.setCustomValidity(""); saveDraft(); } catch { sigma.setCustomValidity("Enter a number, JSON object, or missing indicator for sigma."); } };
    const remove = element("button", { type: "button", "aria-label": `Remove observation ${row.id}` }, "Remove");
    remove.onclick = () => removeObservationIds(new Set([row.id]));
    return [row.id, select, input("T_K"), coordinateInput, input("weight"), pin, validation,input("pin_tolerance"), sigma, input("source", "text"), input("group", "text"), remove];
  });
  $("fit-table").replaceChildren(table(["ID", "Kind", "T / K", "Measurements", "Weight", "Hard pin", "Validation only", "Pin tolerance", "σ", "Source", "CV group", ""], editable));
  renderPsat();
}
function removeObservationIds(ids, all = false) {
  observations = observations.filter(row => !ids.has(row.id));
  reconcileFitParameters();
  if (all) {
    observationSets = []; importReports = []; $("fit-input").value = ""; inputDirty = false; pendingImport = null;
    for (const role of new Set(Object.values(manualRoles).flat())) delete manualValues[role];
    renderManual();
  }
  renderObservations(); renderSetOptions(); saveDraft();
}
function clearObservationsPopup() {
  const box = element("div"), select = element("select", { id: "fit-clear-kind" });
  select.append(element("option", { value: "" }, "Choose observations to clear"));
  select.append(element("option", { value: "all" }, `All observations (${observations.length})`));
  for (const kind of catalog.kinds) {
    const count = observations.filter(row => row.kind === kind).length;
    const option = element("option", { value: kind }, `${kind} · ${labels[kind]} (${count})`);
    option.disabled = !count; select.append(option);
  }
  const description = element("p", { role: "status", "aria-live": "polite" }, "Choose one type or all observations. Other observation types are preserved when clearing one type.");
  const clear = element("button", { id: "fit-clear-confirm", type: "button", class: "primary" }, "Clear observations");
  clear.disabled = true;
  select.onchange = () => {
    const all = select.value === "all", count = observations.filter(row => all || row.kind === select.value).length;
    clear.disabled = !select.value || !all && !count;
    clear.textContent = all ? "Clear all observations" : select.value ? `Clear ${select.value} observations` : "Clear observations";
    description.textContent = all ? `Clear all ${count} observations and unfinished pasted/manual input.` : select.value ? `Clear ${count} ${select.value} observations. Keep other types and unfinished input.` : "Choose one type or all observations.";
  };
  clear.onclick = () => {
    const all = select.value === "all", removed = observations.filter(row => all || row.kind === select.value);
    removeObservationIds(new Set(removed.map(row => row.id)), all);
    $("fit-progress").textContent = all ? "Observations cleared." : `Cleared ${removed.length} ${select.value} observations; ${observations.length} remain.`;
    $("modal").close();
  };
  const cancel = element("button", { type: "button" }, "Cancel"); cancel.onclick = () => $("modal").close();
  const actions = element("div", { class: "form-actions" }); actions.append(clear, cancel);
  box.append(labeled("Observation type", select), description, actions); modal("Clear observations", box);
}
const manualRoles = {
  VLE: ["temperature", "pressure", "x1", "y1"], LLE: ["temperature", "x1_alpha", "x1_beta"],
  HE: ["temperature", "x1", "enthalpy"], GAMMA_INF: ["temperature", "gamma1_inf", "gamma2_inf"],
  AZEOTROPE: ["temperature", "pressure", "x1"], UCST: ["temperature", "x1"], LCST: ["temperature", "x1"],
  VLLE:["temperature","pressure","x1_alpha","x1_beta","y1"],
};
function dataKind() { return $("fit-data-kind").value === "AUTO" ? "VLE" : $("fit-data-kind").value; }
function importOptionsForKind(options, kind) {
  const retained = new Set(["table", "column_count", "row_count", "layout", "exclude_rows", "cell_edits"]);
  const result = Object.fromEntries(Object.entries(options).filter(([key]) => retained.has(key)));
  if (kind !== "AUTO") result.kind = kind;
  return result;
}
function vaporPropertiesApplicable() {
  const vaporKinds=["VLE","AZEOTROPE","VLLE"];
  if(observations.some(row=>vaporKinds.includes(row.kind)&&(row.pin||row.validation_only||(row.weight??1)*Number($(`fit-weight-${row.kind}`).value)>0)))return true;
  if($("fit-data-kind").value!=="AUTO")return vaporKinds.includes(dataKind());
  return !observations.length;
}
function renderPsat() {
  const container = $("fit-psat-fields"); container.replaceChildren();
  const vaporApplicable=vaporPropertiesApplicable(),vapor=$("fit-vapor").value;
  const correctionFields=vaporApplicable?(catalog.vapor_requirements[vapor]?.fields||[]):[];
  $("fit-estimate-properties-row").hidden=!correctionFields.length;
  $("fit-hoc-eta-default-row").hidden=!vaporApplicable||vapor!=="HOC";
  for (let index = 0; index < 2; index++) {
    const state = psatValues[index], group = element("div", { class: "structured-group" });
    group.append(element("h3", {}, `Component ${index+1}`));
    const form = choose([["", "PFDSim / imported PFD"], ...Object.entries(catalog.psat_forms).map(([name, definition]) => [name, definition.label])], state.form || "", `fit-psat-${index}-form`);
    form.onchange = () => { state.form = form.value; renderPsat(); saveDraft(); }; group.append(labeled("Psat correlation", form));
    const grid = element("div", { class: "fit-entry-grid" });
    function number(key, label) {
      const input = element("input", { id: `fit-psat-${index}-${key}`, type: "number", step: "any" }); input.value = state[key] ?? "";
      input.oninput = () => { state[key] = input.value; saveDraft(); }; grid.append(labeled(label, input));
    }
    if (state.form) {
      group.append(element("p", { class: "field-help" }, catalog.psat_forms[state.form].equation));
      for (const letter of catalog.psat_forms[state.form].coefficients) number(letter, `Coefficient ${letter}`);
      number("Tmin_K", "Minimum T / K"); number("Tmax_K", "Maximum T / K");
      if (state.form === "antoine") {
        const unit = choose(temperatureUnits.filter(([key]) => key !== "F"), state.temperature_unit || "C", `fit-psat-${index}-temperature_unit`);
        unit.onchange = () => { state.temperature_unit = unit.value; saveDraft(); }; grid.append(labeled("Antoine denominator temperature", unit));
      }
      if (["antoine", "dippr101"].includes(state.form)) {
        const unit = choose(pressureUnits, state.pressure_unit || "bar", `fit-psat-${index}-pressure_unit`);
        unit.onchange = () => { state.pressure_unit = unit.value; saveDraft(); }; grid.append(labeled("Correlation’s absolute pressure unit", unit));
      }
      if (state.form === "canonical_psat_ah") {
        const power = choose([[-3, "−3"], [-5, "−5"], [-7, "−7"]], state.inverse_power || -3, `fit-psat-${index}-inverse_power`);
        power.onchange = () => { state.inverse_power = Number(power.value); saveDraft(); }; grid.append(labeled("Inverse power n", power));
      }
      const source = element("input", { id: `fit-psat-${index}-source`, type: "text" }); source.value = state.source || ""; source.oninput = () => { state.source = source.value; saveDraft(); }; grid.append(labeled("Psat source (optional)", source));
    }
    const required = correctionFields;
    const properties = element("details",{class:"structured-group"});properties.append(element("summary",{},"Optional pure-fluid / vapor properties"));
    const propertyGrid = element("div",{class:"fit-entry-grid"});
    const additional=element("details",{class:"structured-group"});additional.append(element("summary",{},"Additional property overrides"));
    const additionalGrid=element("div",{class:"fit-entry-grid"});
    for (const [key,label] of [["MW","MW / g mol⁻¹"],["Tc_K","Critical temperature / K"],["Pc_bar","Critical pressure / bar"],["Tb_K","Boiling temperature / K"],["omega","Acentric factor"],["Vc_cm3_mol","Critical volume / cm³ mol⁻¹"],["Zc","Critical compressibility"],["dipole_D","Dipole / Debye"],["hoc_eta","HOC pure association η"],["Rprime_A","HOC modified radius / Å"]]) {
      const input=element("input",{id:`fit-psat-${index}-${key}`,type:"number",step:"any",placeholder:"Resolve / estimate if needed"});input.value=state[key]??"";
      const relevant=key==="MW"||required.includes(key)||vaporApplicable&&state.form==="canonical_psat_ah"&&key==="Tc_K";
      input.oninput=()=>{state[key]=input.value;saveDraft();};(relevant?propertyGrid:additionalGrid).append(labeled(label,input));
    }
    const structure=element("input",{id:`fit-psat-${index}-smiles`,type:"text",placeholder:"Optional identifiable structure for estimates"});structure.value=state.smiles||"";structure.oninput=()=>{state.smiles=structure.value;saveDraft();};propertyGrid.append(labeled("SMILES (optional)",structure));
    additional.append(additionalGrid);properties.append(propertyGrid,additional);group.append(grid,properties);container.append(group);
  }
  $("fit-vapor-requirements").textContent = vaporApplicable?(vapor==="IDEAL"?"Psat only · no vapor correction properties.":`${vapor} correction properties`):"Liquid-only data · Psat and vapor corrections are not required.";
}
function readPsat(includePsat = true) {
  includePsat = includePsat && observations.some(row=>["VLE","AZEOTROPE","VLLE"].includes(row.kind));
  const psat = psatValues.map((state, index) => {
    if (!state.form || !includePsat) return null;
    const spec = { form: state.form, coefficients: {}, source: state.source || "", temperature_unit: state.form === "antoine" ? state.temperature_unit || "C" : "K", pressure_unit: ["antoine", "dippr101"].includes(state.form) ? state.pressure_unit || "bar" : "bar" };
    for (const key of [...catalog.psat_forms[state.form].coefficients, "Tmin_K", "Tmax_K"]) {
      if (String(state[key] ?? "").trim() === "") throw new Error(`Component ${index+1} Psat needs ${key}.`);
      const value = Number(state[key]); if (!Number.isFinite(value)) throw new Error(`Component ${index+1} Psat ${key} must be finite.`);
      if (key.length === 1) spec.coefficients[key] = value; else spec[key] = value;
    }
    if (state.form === "canonical_psat_ah") spec.inverse_power = Number(state.inverse_power || -3);
    return spec;
  });
  const component_properties = psatValues.map(state => Object.fromEntries(["MW", "Tc_K", "Pc_bar", "Tb_K", "omega", "Vc_cm3_mol","Zc","dipole_D","hoc_eta","Rprime_A","smiles"].filter(key => String(state[key] ?? "").trim()).map(key => [key, key==="smiles" ? state[key] : Number(state[key])])));
  return { ...(psat.some(Boolean) ? { psat } : {}), ...(component_properties.some(item => Object.keys(item).length) ? { component_properties } : {}) };
}
function choose(items, value, id) {
  const select = element("select", id ? { id } : {});
  for (const [key, label] of items) select.append(element("option", { value: key }, label));
  select.value = value ?? ""; return select;
}
const temperatureUnits = [["C", "°C"], ["K", "K"], ["F", "°F"]];
const pressureUnits = ["bar", "atm", "kpa", "pa", "mpa", "mmhg", "torr", "psi"].map(unit => [unit, unit === "kpa" ? "kPa" : unit === "mpa" ? "MPa" : unit]);
const bases = [["mole_fraction", "Mole fraction (0–1)"], ["mole_percent", "Mole percent (0–100)"], ["mass_fraction", "Mass fraction (0–1)"], ["mass_percent", "Mass percent (0–100)"]];
function componentNames() {
  return [$("fit-comp1").value.trim(), $("fit-comp2").value.trim()].map(identifier => definitionProject?.pfd?.components.find(component => component.symbol === identifier)?.name || identifier);
}
function renderManual() {
  $("fit-paste-controls").hidden = $("fit-input-method").value !== "paste";
  $("fit-manual-controls").hidden = $("fit-input-method").value !== "manual";
  const container = $("fit-manual-fields"); container.replaceChildren();
  const kind = $("fit-data-kind").value === "AUTO" ? "VLE" : $("fit-data-kind").value;
  for (const role of manualRoles[kind]) {
    const input = measurementInput(manualValues[role], { id: `fit-manual-${role}` }, (value, text) => { manualValues[role] = text; saveDraft(); });
    container.append(labeled(catalog.import_fields[role] + (["y1", "gamma1_inf", "gamma2_inf"].includes(role) || ["LLE","VLLE"].includes(kind)&&role.startsWith("x1_") || ["UCST", "LCST"].includes(kind) && role === "x1" ? " (optional)" : ""), input));
  }
  for (const [key, label, items, fallback] of [
    ["temperature_unit", "Temperature unit", temperatureUnits, "C"],
    ["pressure_unit", "Pressure unit", pressureUnits, "bar"],
    ["composition_basis", "Composition basis", bases, "mole_fraction"],
    ["composition_component", "Compositions refer to", componentNames().map((name, index) => [String(index+1), `Component ${index+1} · ${name}`]), "1"],
    ["enthalpy_unit", "Enthalpy unit", enthalpyUnits, "J/mol"],
  ]) {
    if (key === "pressure_unit" && !manualRoles[kind].includes("pressure") || key === "enthalpy_unit" && kind !== "HE" || ["composition_basis", "composition_component"].includes(key) && kind === "GAMMA_INF") continue;
    const input = choose(items, manualValues[key] ?? fallback, `fit-manual-${key}`);
    manualValues[key] = input.value;
    input.onchange = () => { manualValues[key] = input.value; if (key === "composition_basis") renderManual(); saveDraft(); };
    container.append(labeled(label, input));
  }
  if (kind !== "GAMMA_INF" && (manualValues.composition_basis || "").startsWith("mass")) {
    for (let index=0; index<2; index++) {
      const input = element("input", { id: `fit-manual-mw${index+1}`, type: "number", min: 0, step: "any" }); input.value = manualValues[`mw${index+1}`] ?? "";
      input.oninput = () => { manualValues[`mw${index+1}`] = input.value; saveDraft(); };
      container.append(labeled(`Component ${index+1} molecular weight / g mol⁻¹`, input));
    }
  }
  $("fit-manual-validation-only").checked = !!manualValues.validation_only;
}
const sigmaLabels = { log_fugacity: "VLE / LLE σ · log fugacity", log_gamma: "γ∞ σ · log activity coefficient", HE_J_mol: "Hᴱ σ · J/mol", curvature: "Critical curvature σ", third_derivative: "Critical third derivative σ" };
function renderSigma(saved = {}) {
  for (const [key, scale] of Object.entries(catalog.scales)) {
    const input = element("input", { id: `fit-sigma-${key}`, type: "number", min: "0.000000000001", step: "any", placeholder: `Default: ${scale}` }); input.value = saved[key] ?? ""; input.oninput = saveDraft;
    $("fit-set-sigma").append(labeled(sigmaLabels[key], input));
  }
}
function sigmaOverrides() {
  const values = {};
  for (const key of Object.keys(catalog.scales)) {
    const text = $(`fit-sigma-${key}`).value.trim(); if (!text) continue;
    const value = Number(text); if (!Number.isFinite(value) || value <= 0) throw new Error("Uncertainty scales must be positive numbers."); values[key] = value;
  }
  return values;
}
function renderSetOptions() {
  const select = $("fit-sigma-target"), previous = select.value;
  select.replaceChildren(element("option", { value: "all" }, "All observations"));
  for (const set of observationSets) {
    const count = set.ids.filter(id => observations.some(row => row.id === id)).length;
    if (count) select.append(element("option", { value: set.id }, `${set.name} · ${count} observations`));
  }
  select.value = Array.from(select.options).some(option => option.value === previous) ? previous : "all";
  $("fit-apply-sigma").disabled = !observations.length;
}
function addParsed(proposal, mode) {
  const reservedIds = [...observationSets.flatMap(set => set.ids), ...importReports.flatMap(report => report.observation_ids || [])];
  const appended = appendObservations(observations, proposal.observations, sigmaOverrides(), reservedIds);
  observations = appended.rows;
  const id = `set-${observationSets.length+1}`, name = `${mode === "manual" ? "Manual entry" : "Imported table"} ${observationSets.length+1}`;
  if(proposal.series_reports?.length){
    for(const series of proposal.series_reports){
      const seriesId=`${id}-series-${series.index+1}`,ids=series.observation_indices.map(index=>appended.added[index]);
      observationSets.push({id:seriesId,name:`${name} · ${series.name}`,ids});
      for(const index of series.observation_indices)if(proposal.observation_sources[index]?.group_generated){const row=observations.find(item=>item.id===appended.added[index]);row.group=seriesId;}
    }
  }else observationSets.push({ id, name, ids: appended.added });
  importReports.push({ ...proposal, observations: undefined, observation_ids: appended.added, source_ids: appended.sourceIds, name });
  if (mode === "paste") { $("fit-input").value = ""; inputDirty = false; }
  else {
    for (const role of manualRoles[dataKind()]) delete manualValues[role];
    renderManual();
  }
  pendingImport=null;
  renderObservations(); renderSetOptions(); saveDraft();
  $("fit-progress").textContent = `Added ${appended.added.length} observations; ${observations.length} total. Existing data was preserved.`;
}
function importPreview(text, initial, initialOptions) {
  return new Promise(resolve => {
    const box = element("div"), controls = element("div", { class: "fit-entry-grid" }), messages = element("div"), preview = element("div", { class: "fit-import-preview" });
    const mappingContainer = element("div", { class: "fit-entry-grid" }), seriesContainer=element("div"), footer = element("div", { class: "form-actions" });
    const repeated=element("button",{type:"button",id:"fit-import-repeated"},"Repeated series / multicolumn");
    const add = element("button", { type: "button", class: "primary", id: "fit-import-add" }, "Review and add"), cancel = element("button", { type: "button" }, "Cancel");
    let proposal = initial, finished = false, revision = 0;
    function seriesAssignments(value) {
      return value.columns.map(column => {
        if (value.shared_columns?.includes(column.index)) return "shared";
        const group = value.series?.findIndex(spec => spec.columns?.includes(column.index)) ?? -1;
        return group < 0 ? "ignore" : String(group);
      });
    }
    let seriesMode=Array.isArray(initial.series),seriesSpecs=structuredClone(initial.series||[]),assignments=seriesAssignments(initial);
    let seriesInputs=[];
    let seriesPreset=seriesMode ? initial.series_inferred ? "detected" : "custom" : "",singleMapping=initial.columns.map(column=>column.role);
    const options = { ...initial.settings, ...initialOptions, table: initial.table, mapping: initial.columns.map(column => column.role) };
    let cellEdits = new Map((initial.cell_edits || initialOptions.cell_edits || []).map(edit => [`${edit.row}:${edit.column}`, edit]));
    const inputs = {};
    const changedControls = new Set();
    function capture() {
      const result = { ...options, cell_edits: Array.from(cellEdits.values()), mapping: Array.from(mappingContainer.querySelectorAll("select[data-field-role]"), select => select.value) };
      result.infer_series = seriesMode;
      for (const [key, control] of Object.entries(inputs)) {
        if (["pressure", "temperature", "row_count", "column_count", "composition_component", "table"].includes(key)) result[key] = control.value === "" ? null : Number(control.value);
        else result[key] = control.value;
      }
      if (inputs.mw1) { result.molecular_weights = [Number(inputs.mw1.value), Number(inputs.mw2.value)]; delete result.mw1; delete result.mw2; }
      for (const key of ["temperature_unit", "pressure_unit", "enthalpy_unit", "composition_basis", "composition_component"]) {
        if (!(key in initialOptions) && !changedControls.has(key) && proposal.columns.some(column => column.settings?.[key] !== undefined)) delete result[key];
      }
      result.exclude_rows = Array.from(preview.querySelectorAll("input[data-row]"), checkbox => checkbox.checked || checkbox.disabled ? null : Number(checkbox.dataset.row)).filter(index => index !== null);
      if(seriesMode&&seriesSpecs.length){
        const groups=Array.from(mappingContainer.querySelectorAll("select[data-series]"),select=>select.value);
        result.shared_columns=groups.flatMap((group,index)=>group==="shared"?[index]:[]);
        result.series=seriesSpecs.map((spec,index)=>{
          const value={...spec,columns:groups.flatMap((group,column)=>group===String(index)?[column]:[])};
          for(const [key,node] of Object.entries(seriesInputs[index]||{})){
            if(node.value===""){delete value[key];continue;}
            if(node.type==="checkbox"){value[key]=node.checked;continue;}
            value[key]=["temperature","pressure"].includes(key)?Number(node.value):node.value;
          }
          return value;
        });
        result.mapping=result.mapping.map((role,index)=>groups[index]==="ignore"?"ignore":role);
      }else{delete result.series;delete result.shared_columns;}
      const captured = result.kind === proposal.settings.kind ? result : importOptionsForKind(result, result.kind);
      pendingImport={text,options:structuredClone(captured)};saveDraft();return captured;
    }
    async function refresh() {
      const ticket = ++revision;
      add.disabled = true;
      try {
        const updated = await api("/api/fitting/parse", { observations: text, components: componentNames(), import_options: capture() });
        if (ticket !== revision || finished) return;
        proposal = updated; display();
      } catch (error) { if(ticket!==revision||finished)return;add.disabled=true;messages.replaceChildren(element("p", { class: "fit-error" }, error.message)); }
    }
    function renderMappings() {
      mappingContainer.replaceChildren();
      for (const column of proposal.columns) {
        const select = choose(Object.entries(catalog.import_fields), column.role, `fit-import-column-${column.index}`);
        select.dataset.fieldRole="true";select.onchange = guarded(refresh);
        const cell=element("div");cell.append(labeled(column.label,select));
        if(seriesMode){
          const group=choose([["ignore","Ignore this column"],["shared","Shared across all series"],...seriesSpecs.map((spec,index)=>[String(index),spec.name||`Series ${index+1}`])],assignments[column.index]||"ignore",`fit-import-column-series-${column.index}`);
          group.dataset.series=String(column.index);group.onchange=guarded(async()=>{assignments[column.index]=group.value;await refresh();});cell.append(labeled("Applies to",group));
        }
        mappingContainer.append(cell);
      }
    }
    function seriesHints(columns,condition){
      const hints=columns.map(index=>proposal.series_header_hints?.[index]||{}),result={};
      for(const key of ["temperature","temperature_unit","pressure","pressure_unit","enthalpy_unit"]){
        const values=[...new Set(hints.map(hint=>hint[key]).filter(value=>value!==undefined))];if(values.length===1)result[key]=values[0];
      }
      const bare=[...new Set(hints.map(hint=>hint.condition_value).filter(value=>value!==undefined))];if(bare.length===1&&result[condition]===undefined)result[condition]=bare[0];
      return result;
    }
    function presetSeries(preset){
      seriesPreset=preset;
      if(!preset){seriesSpecs=[];renderSeries();add.disabled=true;return;}
      const width=proposal.columns.length,roles=Array(width).fill("ignore");assignments=Array(width).fill("ignore");seriesSpecs=[];
      const create=(columns,condition,defaults={})=>{const index=seriesSpecs.length;seriesSpecs.push({name:`Series ${index+1}`,columns,...seriesHints(columns,condition),...defaults});columns.forEach(column=>assignments[column]=String(index));};
      const detected = proposal.series_suggestion;
      if (detected && (preset === "detected" || preset === "TxyTriples" && detected.layout === "TxyGroups" || preset === "PxyTriples" && detected.layout === "PxyGroups")) {
        inputs.kind.value = "VLE";
        detected.mapping.forEach((role, index) => roles[index] = role);
        const condition = detected.layout === "TxyGroups" ? "pressure" : "temperature";
        for (const spec of detected.series) create(spec.columns, condition, structuredClone(spec));
      }else if(preset==="HE"){
        inputs.kind.value="HE";roles[0]="x1";assignments[0]="shared";
        for(let column=1;column<width;column++){roles[column]="enthalpy";create([column],"temperature");}
      }else if(["Txy","Tx","Pxy","Px"].includes(preset)){
        inputs.kind.value="VLE";roles[0]="x1";assignments[0]="shared";
        const stride=["Tx","Px"].includes(preset)?1:2,measured=preset.startsWith("T")?"temperature":"pressure",condition=measured==="temperature"?"pressure":"temperature";
        for(let column=1;column<width;column+=stride){const columns=[column];roles[column]=measured;if(stride===2&&column+1<width){columns.push(column+1);roles[column+1]="y1";}create(columns,condition);}
      }else if(["TxyTriples", "PxyTriples"].includes(preset)){
        inputs.kind.value="VLE";
        const axis = preset === "TxyTriples" ? "temperature" : "pressure", condition = axis === "temperature" ? "pressure" : "temperature";
        const usable = proposal.columns.filter(column => !["calculated column", "uncertainty column"].includes(column.reason)).map(column => column.index);
        for(let start=0;start<usable.length;start+=3){const columns=usable.slice(start,start+3);columns.forEach((column,offset)=>roles[column]=[axis,"x1","y1"][offset]);create(columns,condition);}
      }else{
        proposal.columns.forEach((column,index)=>roles[index]=column.role);
        seriesSpecs=[{name:"Series 1",columns:[]},{name:"Series 2",columns:[]}];
        proposal.columns.forEach((column,index)=>{assignments[index]=["x1","temperature","pressure"].includes(column.role)&&roles.filter(role=>role===column.role).length===1?"shared":"0";});
      }
      proposal={...proposal,settings:{...proposal.settings,kind:inputs.kind.value},columns:proposal.columns.map((column,index)=>({...column,role:roles[index]}))};
      renderSeries();renderMappings();refresh();
    }
    function renderSeries(){
      seriesContainer.replaceChildren();seriesInputs=[];
      repeated.textContent=seriesMode?"Use single-series mode":"Repeated series / multicolumn";
      if(!seriesMode)return;
      seriesContainer.append(element("h3",{},"Repeated-series layout"),element("p",{class:"field-help"},"Shared columns are reused. Assign each other column to one series, then specify its conditions. Nothing is added until every selected series is ready."));
      const detectedChoice = proposal.series_suggestion ? [["detected", proposal.series_suggestion.layout === "TxyGroups" ? "Recognized temperature / liquid / vapor groups" : "Recognized pressure / liquid / vapor groups"]] : [];
      const layout=choose([["","Choose a layout"],...detectedChoice,["HE","Shared x + HE columns at different temperatures"],["Txy","Shared x + repeated T / vapor pairs"],["Tx","Shared x + temperature columns at different pressures"],["Pxy","Shared x + repeated P / vapor pairs"],["Px","Shared x + pressure columns at different temperatures"],["TxyTriples","Repeated temperature / liquid / vapor triples"],["PxyTriples","Repeated pressure / liquid / vapor triples"],["custom","Custom column groups"]],seriesPreset,"fit-import-series-layout");
      layout.onchange=()=>presetSeries(layout.value);seriesContainer.append(labeled("Layout",layout));
      if(!seriesSpecs.length){add.disabled=true;return;}
      const count=element("input",{id:"fit-import-series-count",type:"number",min:1,max:30,step:1});count.value=seriesSpecs.length;
      count.onchange=guarded(async()=>{const n=Number(count.value);if(!Number.isInteger(n)||n<1||n>30)throw new Error("Use 1–30 series.");const old=capture().series;seriesSpecs=Array.from({length:n},(_,index)=>old[index]||{name:`Series ${index+1}`,columns:[]});assignments=assignments.map(group=>group==="shared"||group==="ignore"||Number(group)<n?group:"ignore");renderSeries();renderMappings();await refresh();});
      seriesContainer.append(labeled("Number of series",count));
      seriesSpecs.forEach((spec,index)=>{
        const row=element("div",{class:"structured-group"}),grid=element("div",{class:"fit-entry-grid"});row.append(element("h4",{},`Series ${index+1}`));const nodes={};seriesInputs.push(nodes);
        function field(key,label,choices){const node=choices?choose(choices,spec[key]??"",`fit-import-series-${index}-${key}`):element("input",{id:`fit-import-series-${index}-${key}`,type:key==="name"?"text":"number",step:"any"});if(!choices)node.value=spec[key]??"";nodes[key]=node;node.oninput=()=>{++revision;};node.onchange=guarded(refresh);grid.append(labeled(label,node));}
        field("name","Name");
        field("temperature","Temperature · if not in a data column");field("temperature_unit","Temperature unit",[["","Use common unit"],...temperatureUnits]);
        if(["VLE","VLLE","AZEOTROPE","LLE"].includes(inputs.kind.value)){field("pressure","Pressure · if not in a data column");field("pressure_unit","Pressure unit",[["","Use common unit"],...pressureUnits]);}
        if(inputs.kind.value==="HE")field("enthalpy_unit","HE unit",[["","Use common unit"],...enthalpyUnits]);
        const validation=element("input",{id:`fit-import-series-${index}-validation_only`,type:"checkbox"});validation.checked=!!spec.validation_only;validation.onchange=guarded(refresh);nodes.validation_only=validation;
        const label=element("label",{class:"fit-checkbox-row"});label.append(validation,element("span",{},"Validation-only series"));row.append(grid,label);seriesContainer.append(row);
      });
    }
    repeated.onclick=()=>{
      if(seriesMode){seriesMode=false;seriesSpecs=[];proposal={...proposal,columns:proposal.columns.map((column,index)=>({...column,role:singleMapping[index]||"ignore"}))};renderSeries();renderMappings();refresh();return;}
      seriesMode=true;singleMapping=Array.from(mappingContainer.querySelectorAll("select[data-field-role]"),select=>select.value);
      const roles=singleMapping;
      if(proposal.series_suggestion)presetSeries("detected");
      else if(inputs.kind.value==="HE")presetSeries("HE");
      else if(roles.filter(role=>role==="temperature").length>1&&roles.filter(role=>role==="x1").length===1)presetSeries(roles.filter(role=>role==="y1").length>1?"Txy":"Tx");
      else{seriesSpecs=[];renderSeries();renderMappings();messages.replaceChildren(element("p",{class:"field-help"},"Choose a repeated-series layout and fill its conditions."));add.disabled=true;}
    };
    async function reshape(newOptions) {
      const ticket = ++revision; add.disabled = true;
      const updated = await api("/api/fitting/parse", { observations: text, components: componentNames(), import_options: newOptions });
      if (ticket !== revision || finished) return;
      proposal = updated;
      cellEdits = new Map((updated.cell_edits || []).map(edit => [`${edit.row}:${edit.column}`, edit]));
      seriesMode=Array.isArray(updated.series);seriesSpecs=structuredClone(updated.series||[]);assignments=seriesAssignments(updated);seriesPreset=seriesMode ? updated.series_inferred ? "detected" : "custom" : "";
      for (const key of Object.keys(options)) delete options[key];
      Object.assign(options, updated.settings, { table: updated.table });
      for (const [key, node] of Object.entries(inputs)) {
        if (key === "table") node.value = String(updated.table);
        else if (key === "column_count") node.value = updated.column_count ?? (updated.dimensions_needed ? "" : updated.columns.length);
        else if (key === "row_count") node.value = updated.row_count ?? "";
        else if (key === "layout") node.value = updated.layout;
        else if (key === "mw1" || key === "mw2") node.value = updated.settings.molecular_weights?.[key === "mw1" ? 0 : 1] ?? "";
        else node.value = updated.settings[key] ?? "";
      }
      renderSeries(); renderMappings(); display();
    }
    function display() {
      messages.replaceChildren(...proposal.notes.map(note => element("p", { class: "field-help" }, note)), ...proposal.issues.map(issue => element("p", { class: "fit-error" }, issue)));
      const hasAmbiguousRows=!!proposal.ambiguous_rows?.length;
      preview.replaceChildren(table(["Include", ...proposal.columns.map(column => column.label),...(hasAmbiguousRows?["Column alignment"]:[])], proposal.raw_rows.map((cells, index) => {
        const checkbox = element("input", { type: "checkbox", "data-row": index, "aria-label": `Include source row ${index+1}` });
        const ambiguous=proposal.ambiguous_rows?.find(item=>item.row===index);
        const excluded = proposal.excluded.find(item => item.row === index && (!seriesMode || proposal.series_reports?.every(series=>series.excluded?.some(exclusion=>exclusion.row===index))));
        checkbox.checked = !excluded || !!ambiguous&&excluded.reason!=="Excluded in the import preview."; checkbox.disabled = !!excluded?.reason.startsWith("Pure-component"); checkbox.title = ambiguous?.reason || excluded?.reason || "Include this source row";
        const shown = cells.map((cell, column) => {
          const node = element("input", { type: "text", "aria-label": `Source row ${index+1} column ${column+1}`, maxlength: 1000, placeholder: "Blank / ?" });
          node.disabled = !!proposal.dimensions_needed;
          node.value = cellEdits.get(`${index}:${column}`)?.value ?? (ambiguous && column > 0 ? "" : cell);
          node.oninput = () => {
            ++revision; add.disabled = true;
            if (!ambiguous) { cellEdits.set(`${index}:${column}`, { row: index, column, value: node.value }); capture(); }
          };
          if (!ambiguous) node.onchange = guarded(refresh);
          return node;
        });
        let alignment = "Aligned";
        if (ambiguous) {
          alignment = element("div");
          alignment.append(element("span", { class: "fit-error" }, `Unassigned values: ${ambiguous.values.join(", ")}`));
          const confirm = element("button", { type: "button", "aria-label": `Confirm alignment of source row ${index+1}` }, "Confirm edited row");
          confirm.onclick = guarded(async () => {
            shown.forEach((node, column) => cellEdits.set(`${index}:${column}`, { row: index, column, value: node.value }));
            await refresh();
          });
          alignment.append(confirm);
        }
        checkbox.onchange = guarded(refresh); return [checkbox, ...shown,...(hasAmbiguousRows?[alignment]:[])];
      })));
      add.disabled = !proposal.ready; add.textContent = proposal.ready ? `Add ${proposal.observations.length} observations` : "Resolve the missing choices";
      if(proposal.series_reports?.length)messages.append(element("p",{class:"field-help"},proposal.series_reports.map(series=>`${series.name}: ${series.ready?series.observations+" observations":"conditions needed"}`).join(" · ")));
    }
    function control(key, label, choices, value) {
      const node = choices ? choose(choices, value, `fit-import-${key}`) : element("input", { id: `fit-import-${key}`, type: "number", step: "any" });
      if (!choices) node.value = value ?? "";
      inputs[key] = node; node.oninput = () => { ++revision; }; node.onchange = guarded(() => { changedControls.add(key); return refresh(); }); controls.append(labeled(label, node)); return node;
    }
    const kindControl = control("kind", "Observation kind", catalog.kinds.map(kind => [kind, labels[kind]]), options.kind);
    kindControl.onchange = guarded(async () => {
      changedControls.clear();
      await reshape(importOptionsForKind(capture(), kindControl.value));
    });
    control("temperature_unit", "Temperature unit · confirm if suggested", temperatureUnits, options.temperature_unit);
    control("temperature", "Common temperature (if absent from rows)", null, options.temperature);
    control("pressure_unit", "Pressure unit", [["", "Choose if unknown"], ...pressureUnits], options.pressure_unit || "");
    control("pressure", "Common pressure (if absent from rows)", null, options.pressure);
    control("composition_basis", "Composition basis · confirm if suggested", bases, options.composition_basis || "mole_fraction");
    control("composition_component", "Compositions describe", [["", "Choose the component"], ...componentNames().map((name, i) => [String(i+1), `Component ${i+1} · ${name}`])], options.composition_component);
    control("enthalpy_unit", "Excess-enthalpy unit", [["", "Choose if needed"], ...enthalpyUnits], options.enthalpy_unit);
    control("mw1", "Component 1 molecular weight · if using mass basis", null, options.molecular_weights?.[0]);
    control("mw2", "Component 2 molecular weight · if using mass basis", null, options.molecular_weights?.[1]);
    if (proposal.tables.length > 1) {
      const selectedTable = control("table", "Table from this paste", proposal.tables.map(item => [String(item.index), `Table ${item.index+1} · ${item.rows} rows, ${item.columns} columns`]), String(proposal.table));
      selectedTable.onchange = guarded(async () => {
        await reshape({ table: Number(selectedTable.value) });
      });
    }
    if (proposal.flattened) {
      const count = control("column_count", "Columns in flattened PDF text", null, proposal.column_count ?? (proposal.dimensions_needed ? "" : proposal.columns.length));
      const rows = control("row_count", "Rows in flattened PDF text (data rows only)", null, proposal.row_count);
      const layout = control("layout", "Flattened text order", [["rows", "Read across rows"], ["columns", "Read down columns"]], proposal.layout);
      for (const node of [count, rows, layout]) node.onchange = guarded(async () => {
        const newOptions = capture(); delete newOptions.mapping; delete newOptions.cell_edits; newOptions.exclude_rows = []; await reshape(newOptions);
      });
    }
    renderMappings();renderSeries();
    box.append(element("p", {}, `Review the interpretation and edit cells as needed. Blank, ?, None, N/A, dashes, and other missing indicators mean absent data. Ambiguous rows require an explicit column alignment. Adding this table preserves the ${observations.length} observations already entered.`));
    if (proposal.reference_component) {
      box.append(element("p", {}, `The source labels these compositions as ${proposal.reference_component}.`));
      if (!definitionProject && !componentNames().some(name => name.toLowerCase() === proposal.reference_component.toLowerCase())) {
        if (observations.length) box.append(element("p", { class: "field-help" }, "Changing the mixture components also changes the interpretation of observations already entered. Clear unrelated data before combining measurements for another mixture."));
        const use = element("button", { type: "button" }, `Use ${proposal.reference_component} as component 1`);
        use.onclick = guarded(async () => { $("fit-comp1").value = proposal.reference_component; inputs.composition_component.options[1].textContent = `Component 1 · ${proposal.reference_component}`; inputs.composition_component.value = "1"; renderManual(); saveDraft(); await refresh(); });
        box.append(use);
      }
    }
    repeated.hidden=proposal.series_available===false;
    footer.append(add, cancel); box.append(controls,repeated,seriesContainer,element("h3", {}, "Column meanings"), mappingContainer, messages, preview, footer);
    const dialog = $("modal"), closeButton = $("modal-close");
    function finishPreview(value, preserveDraft = false) {
      if (finished) return;
      if (preserveDraft) capture();
      finished = true;
      dialog.removeEventListener("close", dismissPreview);
      dialog.removeEventListener("cancel", dismissPreview);
      closeButton.removeEventListener("click", dismissPreview);
      resolve(value);
    }
    function dismissPreview() { finishPreview(null, true); }
    cancel.onclick = () => { dismissPreview(); dialog.close(); };
    add.onclick = guarded(async () => {
      // Revalidate the current controls, including edits that have not blurred.
      ++revision; add.disabled = true;
      const updated = await api("/api/fitting/parse", { observations: text, components: componentNames(), import_options: capture() });
      proposal = updated; display(); if (!proposal.ready) return;
      finishPreview(proposal); dialog.close();
    });
    dialog.addEventListener("close", dismissPreview);
    dialog.addEventListener("cancel", dismissPreview);
    closeButton.addEventListener("click", dismissPreview);
    modal("Interpret pasted observations", box); display();
  });
}
async function parseInput() {
  const mode = $("fit-input-method").value;
  let text = $("fit-input").value, options = $("fit-data-kind").value === "AUTO" ? {} : {kind:dataKind()};
  if(mode==="paste"&&pendingImport?.text===text)options={...pendingImport.options,...options};
  if (mode === "manual") {
      const kind = $("fit-data-kind").value === "AUTO" ? "VLE" : $("fit-data-kind").value;
    const roles = manualRoles[kind].filter(role => !missingValue(manualValues[role]));
    if (!roles.length) throw new Error("Enter an observation in the textboxes first.");
    const required = { VLE: ["temperature", "pressure", "x1"], LLE: ["temperature"], HE: ["temperature", "x1", "enthalpy"], GAMMA_INF: ["temperature"], AZEOTROPE: ["temperature", "pressure", "x1"], VLLE:["temperature","pressure"], UCST: ["temperature"], LCST: ["temperature"] }[kind];
    const missing = required.filter(role => !roles.includes(role));
    if (kind === "GAMMA_INF" && !roles.some(role => role.startsWith("gamma"))) missing.push("gamma1_inf or gamma2_inf");
    if (kind === "LLE" && !roles.some(role => ["x1_alpha", "x1_beta"].includes(role))) missing.push("x1_alpha or x1_beta");
    if (missing.length) throw new Error(`Fill the missing observation fields: ${missing.join(", ")}.`);
    text = roles.join("\t")+"\n"+roles.map(role => manualValues[role]).join("\t");
    options = { kind, mapping: roles, temperature_unit: manualValues.temperature_unit, pressure_unit: manualValues.pressure_unit || "bar", composition_basis: manualValues.composition_basis || "mole_fraction", composition_component: kind === "GAMMA_INF" ? 1 : Number(manualValues.composition_component), enthalpy_unit: manualValues.enthalpy_unit || "J/mol" };
    if ((manualValues.composition_basis || "").startsWith("mass")) options.molecular_weights = [Number(manualValues.mw1), Number(manualValues.mw2)];
  }
  if (!text.trim()) throw new Error("Paste a table or enter an observation first.");
  let parsed;
  try{parsed=await api("/api/fitting/parse",{observations:text,components:componentNames(),import_options:options});}
  catch(error){
    if(mode!=="paste"||pendingImport?.text!==text||error.status!==400)throw error;
    const base=await api("/api/fitting/parse",{observations:text,components:componentNames(),import_options:$("fit-data-kind").value==="AUTO"?{}:{kind:dataKind()}});
    parsed={...base,ready:false,needs_review:true,issues:[error.message],settings:{...base.settings,...Object.fromEntries(Object.entries(options).filter(([key])=>!["series","shared_columns","mapping","table","exclude_rows","column_count","layout"].includes(key)))},
            columns:base.columns.map((column,index)=>({...column,role:options.mapping?.[index]||column.role})),series:options.series,shared_columns:options.shared_columns};
  }
  if (mode === "manual" && !parsed.ready) throw new Error(parsed.issues.join(" "));
  if (mode === "paste" && (!parsed.ready || parsed.needs_review)) parsed = await importPreview(text, parsed, options);
  if (!parsed) return false;
  if(mode==="manual")for(const row of parsed.observations)row.validation_only=$("fit-manual-validation-only").checked;
  addParsed(parsed, mode); return true;
}
function scopes(select, project) {
  select.replaceChildren(element("option", { value: "global" }, "Global"));
  for (const scope of project?.pfd?.thermo_scopes || []) select.append(element("option", { value: scope.name }, scope.name));
}
function libraryOptions() {
  const library = readLocal("pfdsim.laboratories.v1", {});
  for (const [id, placeholder, project] of [["fit-project", "Independent mixture", definitionProject], ["fit-export-project", "Choose a laboratory", exportProject]]) {
    $(id).replaceChildren(element("option", { value: "" }, placeholder));
    for (const project of Object.values(library)) $(id).append(element("option", { value: project.id }, project.pfd?.metadata?.process_name || project.filename || project.id));
    const selected = projectChoice(project);
    if (project && !Array.from($(id).options).some(option => option.value === selected)) {
      $(id).append(element("option", { value: selected }, project.pfd?.metadata?.process_name || project.filename || "Imported PFD"));
    }
    $(id).value = selected;
  }
}
function projectChoice(project) { return project ? project.id || "@imported-pfd" : ""; }
function selectedProject(select, current) {
  if (current && select.value === projectChoice(current)) return current;
  return readLocal("pfdsim.laboratories.v1", {})[select.value] || null;
}
async function selectDefinition(project) {
  definitionProject = project;
  libraryOptions();
  scopes($("fit-scope"), project);
  if (project?.pfd?.components?.length >= 2) {
    [$("fit-comp1").value, $("fit-comp2").value] = project.pfd.components.slice(0, 2).map(c => c.symbol);
  }
  renderManual(); saveDraft();
}
function exportMapping() {
  libraryOptions();
  scopes($("fit-export-scope"), exportProject);
  const container = $("fit-export-map"); container.replaceChildren();
  if (!exportProject || !result) return;
  result.components.forEach((symbol, index) => {
    const label = element("label", { class: "field" }); label.append(element("span", { class: "field-label" }, `Map ${result.component_names[index]} to`));
    const select = element("select", { id: `fit-map-${index}` });
    for (const component of exportProject.pfd.components) select.append(element("option", { value: component.symbol }, `${component.symbol} · ${component.name}`));
    const original = result.request.components[index];
    const match = exportProject.pfd.components.find(c => c.symbol === original || c.name.toLowerCase() === original.toLowerCase());
    if (match) select.value = match.symbol;
    label.append(select); container.append(label);
  });
}
function residualPlot(points) {
  const svg = svgElement("svg", { viewBox: "0 0 760 260", role: "img", "aria-label": "Scaled residuals by observation. Zero indicates agreement." });
  const max = Math.max(1, ...points.flatMap(point => point.scaled_residuals.map(Math.abs)));
  svg.append(svgElement("path", { d: "M45 120H735", stroke: "#698696" }));
  const zero = svgElement("text", { x: 10, y: 125, fill: "#698696" }); zero.textContent = "0"; svg.append(zero);
  const title = svgElement("text", { x: 45, y: 20, fill: "currentColor" }); title.textContent = `Scaled residuals · vertical range ±${formatNumber(max)}`; svg.append(title);
  points.forEach((point, index) => {
    const x = 50+index*680/Math.max(1, points.length-1);
    for (const residual of point.scaled_residuals) {
      const circle = svgElement("circle", { cx: x, cy: 120-residual/max*80, r: 3, fill: point.physical ? "#78e6cb" : "#ff8996" });
      const tooltip = svgElement("title"); tooltip.textContent = `${point.id} · ${point.kind}: ${formatNumber(residual)}`; circle.append(tooltip); svg.append(circle);
    }
  }); return svg;
}
function renderResult() {
  if (!result) return;
  $("fit-results").hidden = false;
  $("fit-result-status").textContent = `${result.component_names?.join(" / ")||result.components.join(" / ")} · ${result.success ? "Converged; training phase checks passed" : "Fit needs review"} · ${result.method} · ${result.points.filter(point=>point.role!=="validation").length} training / ${result.points.filter(point=>point.role==="validation").length} validation-only observations`;
  $("fit-warnings").replaceChildren(...result.warnings.map(message => element("p", { class: "fit-error" }, message)));
  $("fit-objectives").replaceChildren(table(["Objective", "Points", "Scaled RMSE", "Max |scaled error|", "Weighted SSE"], Object.entries(result.objectives).map(([kind, score]) => [kind, score.points, formatNumber(score.scaled_RMSE), formatNumber(score.scaled_max_abs), formatNumber(score.weighted_sum_squares)])));
  const metrics=[];
  for(const [role,groups] of [["Training",result.physical_metrics||{}],["Validation-only",result.validation_only?.physical_metrics||{}]])for(const [kind,quantities] of Object.entries(groups))for(const [quantity,record] of Object.entries(quantities))metrics.push([role,kind,quantity,record.unit,record.n,formatNumber(record.MAE),formatNumber(record.RMSE),formatNumber(record.bias),record.unavailable]);
  $("fit-physical-metrics").replaceChildren(table(["Data use","Objective","Quantity","Unit","N","MAE","RMSE","Bias","Unavailable"],metrics));
  renderObjectivePlots($("fit-objective-plots"),result.plots);
  $("fit-plot").replaceChildren(residualPlot(result.points));
  function predictionText(point) {
    const p = point.predicted;
    if (p.equilibrium_error) return p.equilibrium_error;
    if (point.kind === "HE") return `Hᴱ = ${formatNumber(p.HE_J_mol)} J/mol`;
    if (point.kind === "GAMMA_INF") return [p.gamma1_inf !== undefined ? `γ₁∞ = ${formatNumber(p.gamma1_inf)}` : "", p.gamma2_inf !== undefined ? `γ₂∞ = ${formatNumber(p.gamma2_inf)}` : ""].filter(Boolean).join("; ");
    if (["VLE", "AZEOTROPE"].includes(point.kind)) return `P = ${formatNumber(p.P_bar)} bar; y₁ = ${formatNumber(p.y1)}`;
    if (["LLE","VLLE"].includes(point.kind)) return `x₁α = ${formatNumber(p.x1_alpha)}; x₁β = ${formatNumber(p.x1_beta)}${point.kind==="VLLE"?`; y₁ = ${formatNumber(p.y1)}; P = ${formatNumber(p.P_bar)} bar`: `; split ${p.split ? "present" : "absent"}`}`;
    return `x₁ = ${formatNumber(p.x1)}; g″ = ${formatNumber(p.curvature)}; g‴ = ${formatNumber(p.third_derivative)}`;
  }
  const physicalErrors=point=>Object.entries({T_K:"K",P_bar:"bar",y1:"mole fraction",x1_alpha:"mole fraction",x1_beta:"mole fraction",HE_J_mol:"J/mol",gamma1_inf:"",gamma2_inf:"",x1:"mole fraction"}).filter(([key])=>Number.isFinite(point.observed[key])&&Number.isFinite(point.predicted[key])&&(key!=="x1"||["UCST","LCST"].includes(point.kind))).map(([key,unit])=>`${key}: ${formatNumber(point.predicted[key]-point.observed[key])} ${unit}`).join("; ")||"Unavailable / not measured";
  $("fit-points").replaceChildren(table(["Point","Kind","Data use","Predictions","Physical errors (predicted − observed)","Phase checks","Pin"],result.points.map(point=>[point.id,point.kind,point.role||"training",predictionText(point),physicalErrors(point),point.physical?"Passed":"Failed",point.pin_satisfied===null?"—":point.pin_satisfied?"Satisfied":"Failed"])));
  $("fit-diagnostic-points").replaceChildren(table(["Point","Kind","Scaled residuals","Weighted SSE"],result.points.map(point=>[point.id,point.kind,point.scaled_residuals.map(value=>formatNumber(value)).join(", "),formatNumber(point.weighted_sum_squares)])));
  $("fit-validation").replaceChildren(element("p", {}, result.cross_validation.pinned_points_policy), table(["Fold", "Fit converged", "Rank", "Held-out observations", "Held-out scores / errors"], result.cross_validation.folds.map(fold => [fold.fold, fold.success ? "Yes" : "No", fold.rank === undefined ? "—" : `${fold.rank}/${fold.parameter_count}`, fold.held_out_ids.join(", "), fold.error || JSON.stringify(fold.objectives)])));
  $("fit-coefficients").textContent = JSON.stringify({ coefficients: result.coefficients, parameters: result.parameters, vapor_parameters: result.vapor_parameters, rq: result.rq, optimizer: result.optimizer,property_provenance:result.property_provenance }, null, 2);
  $("fit-entry").value = result.entry;
  $("fit-submit").disabled = false;
  $("fit-admin-direct").disabled = false;
  exportMapping();
}
async function followJob(identifier, kind = "fit") {
  activeJob = identifier; $("fit-run").disabled = true; $("fit-prefill").disabled = true; $("fit-cancel").disabled = false; saveDraft();
  try {
    const job = await pollJob(identifier, update => { $("fit-progress").textContent = update.progress.at(-1) || update.status; });
    if (job.status !== "completed") throw new Error(job.error || `Calculation ${job.status}.`);
    if (kind === "fit_prefill") {
      if (JSON.stringify(job.output.identifiers) !== JSON.stringify([$("fit-comp1").value.trim(), $("fit-comp2").value.trim()]))
        throw new Error("Components changed while R/Q was being resolved. Prefill again for the current components.");
      [$("fit-r1").value, $("fit-q1").value, $("fit-r2").value, $("fit-q2").value] = job.output.rq.flatMap(item => [item.r, item.q]);
      job.output.warnings.forEach(message => toast(message));
    } else { result = job.output; jobId = identifier; renderResult(); }
    $("fit-progress").textContent = kind === "fit_prefill" ? "R/Q prefilled." : result.success ? "Fit completed. Review the assessment and export below." : "Calculation completed; the fit needs review.";
  } finally { activeJob = null; $("fit-run").disabled = false; $("fit-prefill").disabled = false; $("fit-cancel").disabled = true; saveDraft(); }
}
async function exportMerged() {
  if (!exportProject) throw new Error("Choose a laboratory or import a PFD for export.");
  if (exportProject.pending) throw new Error("Apply or discard this laboratory’s text draft in Workspace before exporting a fit into it.");
  return api("/api/fitting/export", { result, pfd_text: exportProject.text, scope: $("fit-export-scope").value, component_map: Object.fromEntries(result.components.map((symbol, i) => [symbol, $(`fit-map-${i}`).value])) });
}
async function importPfd(file, destination) {
  const parsed = await api("/api/parse", { text: await file.text() });
  const project = { id: null, text: parsed.text, pfd: parsed.pfd, filename: file.name };
  if (destination === "definition") await selectDefinition(project);
  else { exportProject = project; exportMapping(); saveDraft(); }
}

function currentSource() {
  return { citation: $("fit-source").value.trim(), url: $("fit-source-url").value.trim(), doi: $("fit-source-doi").value.trim(), notes: $("fit-source-notes").value.trim() };
}
function assignFitState(saved){
  saved=structuredClone(saved);
  for(const [name,value] of Object.entries(saved.controls||{})){const control=$(`fit-${name}`);if(control){if(control.type==="checkbox")control.checked=value;else control.value=value;}}
  for(const [kind,value] of Object.entries(saved.weights||{}))if($(`fit-weight-${kind}`))$(`fit-weight-${kind}`).value=value;
  observations=saved.observations||[];inputDirty=saved.inputDirty;result=saved.result||null;jobId=saved.jobId||null;definitionProject=saved.definitionProject||null;exportProject=saved.exportProject||null;
  observationSets=saved.observationSets||[];importReports=saved.importReports||[];manualValues=saved.manualValues||{};psatValues=saved.psatValues||[{},{}];pendingImport=saved.pendingImport||null;
  parameterModel=$("fit-model").value;
  return saved;
}
function repairFitObservationReferences(){
  const repaired=repairObservationIds(observations);observations=repaired.rows;
  for(const set of observationSets)set.ids=set.ids.map(id=>repaired.renames[id]||id);
  for(const report of importReports){
    if(report.observation_ids)report.observation_ids=report.observation_ids.map(id=>repaired.renames[id]||id);
    if(report.source_ids)for(const entry of report.source_ids)entry.observation_id=repaired.renames[entry.observation_id]||entry.observation_id;
  }
  for(const field of ["initial","bounds"]){
    const node=$(`fit-${field}`);if(!node.value.trim())continue;
    try{const parsed=JSON.parse(node.value);if(!parsed||Array.isArray(parsed)||typeof parsed!=="object")continue;
      let changed=false;
      const entries=Object.entries(parsed).map(([key,value])=>{const dot=key.indexOf("."),prefix=key.slice(0,dot),id=key.slice(dot+1);if(["critical_x1","vlle_xa","vlle_gap"].includes(prefix)&&repaired.renames[id]){changed=true;key=`${prefix}.${repaired.renames[id]}`;}return [key,value];});
      if(changed)node.value=JSON.stringify(Object.fromEntries(entries),null,2);
    }catch{/* Preserve unfinished JSON as an editable draft. */}
  }
}
function restoreFitState(state){
  const saved=assignFitState(state);
  libraryOptions();
  for(const key of Object.keys(catalog.scales))$(`fit-sigma-${key}`).value=saved.sigmaValues?.[key]??"";
  repairFitObservationReferences();
  renderManual();renderSetOptions();renderPsat();renderVaporParameters(saved.vaporParameters||[]);reconcileFitParameters();updateModel();renderObservations();
  $("fit-results").hidden=!result;renderResult();scopes($("fit-scope"),definitionProject);
  if(saved.selectedScope)$("fit-scope").value=saved.selectedScope;
  if(saved.exportScope)$("fit-export-scope").value=saved.exportScope;
  if(result)result.components.forEach((name,index)=>{if(saved.exportMapping?.[name]&&$(`fit-map-${index}`))$(`fit-map-${index}`).value=saved.exportMapping[name];});
  activeJob=null;$("fit-run").disabled=false;$("fit-cancel").disabled=true;
  $("fit-progress").textContent=`Restored ${observations.length} observations and their fitting settings. No calculation was rerun.`;saveDraft();
}

async function initialize() {
  catalog = await api("/api/fitting/catalog");
  for (const vapor of catalog.vapors) $("fit-vapor").append(element("option", { value: vapor }, vapor));
  for (const form of Object.keys(catalog.forms)) $("fit-law").append(element("option", { value: form }, laws[form]));
  $("fit-law").value = "constant_inverse";
  $("fit-scales").value = JSON.stringify(catalog.scales, null, 2);
  for (const kind of catalog.kinds) {
    const label = element("label", { class: "field" }); label.append(element("span", { class: "field-label" }, labels[kind]));
    const input = element("input", { id: `fit-weight-${kind}`, type: "number", min: 0, max: 1e12, step: "any", value: "1" }); input.onchange = ()=>{reconcileFitParameters();renderPsat();saveDraft();}; label.append(input); $("fit-weights").append(label);
    $("fit-data-kind").append(element("option", { value: kind }, labels[kind]));
  }
  let saved = readLocal(draftKey, null);
  if (saved) {
    saved=assignFitState(saved);
  }
  repairFitObservationReferences();
  renderSigma(saved?.sigmaValues); renderManual(); renderSetOptions(); renderPsat();
  renderVaporParameters(saved?.vaporParameters); const reconciledParameters = reconcileFitParameters(); updateModel(); parameterModel=$("fit-model").value; renderObservations(); renderResult(); scopes($("fit-scope"), definitionProject);
  if(saved?.selectedScope)$("fit-scope").value=saved.selectedScope;
  if(saved?.exportScope)$("fit-export-scope").value=saved.exportScope;
  if(result)result.components.forEach((name,index)=>{if(saved?.exportMapping?.[name]&&$(`fit-map-${index}`))$(`fit-map-${index}`).value=saved.exportMapping[name];});
  if (reconciledParameters) saveDraft();
  fields.filter(name => name !== "law").forEach(name => $(`fit-${name}`).addEventListener("change", () => { reconcileFitParameters(); saveDraft(); }));
  $("fit-model").addEventListener("change", () => { reconcileModelChange(); reconcileFitParameters(); updateModel(); saveDraft(); });
  $("fit-law").addEventListener("change", () => { reconcileFitParameters(); updateModel(); saveDraft(); });
  $("fit-vapor").addEventListener("change", () => { renderVaporParameters(); renderPsat();saveDraft(); });
  $("fit-input").addEventListener("input", () => { pendingImport = null; inputDirty = true; saveDraft(); });
  $("fit-parse").onclick = guarded(parseInput);
  sessionLibrary=fittingSessionLibrary({capture:captureFitState,restore:restoreFitState,canOpen:()=>!activeJob});
  $("fit-saved-sessions").onclick=guarded(()=>sessionLibrary.open());
  $("fit-help").onclick = () => {const content=element("div");content.append($("fit-help-content").content.cloneNode(true));modal("Input fields, uncertainty and validation",content);};
  $("fit-property-help").onclick = () => {const content=element("div");content.append($("fit-property-help-content").content.cloneNode(true));modal("Psat and supporting-property help",content);};
  $("fit-set-uncertainty").onclick = () => {
    const content=$("fit-sigma-controls");content.hidden=false;
    const done=element("button",{type:"button"},"Done");done.onclick=()=>$("modal").close();content.append(done);
    $("modal").addEventListener("close",()=>{done.remove();content.hidden=true;$("fit-sigma-home").append(content);},{once:true});
    modal("Set uncertainty for observations",content);
  };
  $("fit-reset").onclick = () => {
    const defaults={model:"NRTL",vapor:"IDEAL",law:"constant_inverse",alpha:"0.3","free-alpha":false,r1:"",q1:"",r2:"",q2:"",cv:"none",folds:"5",tref:"298.15",starts:"3",evaluations:"500",seed:"1729",extrapolation:"unrestricted",online:false,initial:"",bounds:"","estimate-properties":true,"hoc-eta-default":true};
    for(const [key,value] of Object.entries(defaults)){const node=$(`fit-${key}`);if(node.type==="checkbox")node.checked=value;else node.value=value;}
    for(const kind of catalog.kinds)$(`fit-weight-${kind}`).value="1";
    for(const key of Object.keys(catalog.scales))$(`fit-sigma-${key}`).value="";
    $("fit-scales").value=JSON.stringify(catalog.scales,null,2);psatValues=[{},{}];renderPsat();renderVaporParameters();updateModel();saveDraft();
    toast("Fitting settings reset to defaults. Observations, component identities and completed results were retained.");
  };
  $("fit-manual-validation-only").onchange = ()=>{manualValues.validation_only=$("fit-manual-validation-only").checked;saveDraft();};
  $("fit-input-method").addEventListener("change", () => {if($("fit-input-method").value==="manual"&&$("fit-data-kind").value==="AUTO")$("fit-data-kind").value="VLE";renderManual();renderPsat();saveDraft();});
  $("fit-data-kind").addEventListener("change", () => {
    if (pendingImport) {
      pendingImport = { ...pendingImport, options: importOptionsForKind(pendingImport.options, $("fit-data-kind").value) };
      $("fit-progress").textContent = "Data type changed; review column assignments and conditions again. Cell edits are preserved.";
    }
    renderManual(); renderPsat(); saveDraft();
  });
  $("fit-clear").onclick = clearObservationsPopup;
  $("fit-apply-sigma").onclick = guarded(() => {
    const sigma = sigmaOverrides(); if (!Object.keys(sigma).length) throw new Error("Fill at least one uncertainty scale first.");
    const target = $("fit-sigma-target").value;
    const ids = target === "all" ? observations.map(row => row.id) : observationSets.find(set => set.id === target)?.ids || [];
    for (const row of observations) if (ids.includes(row.id)) row.sigma = mergeSigma(row.sigma, sigma);
    renderObservations(); saveDraft(); toast(`Updated σ for ${observations.filter(row => ids.includes(row.id)).length} observations.`);
  });
  $("fit-example").onclick = () => { $("fit-input-method").value="paste";renderManual();$("fit-input").value = JSON.stringify([exampleRows[dataKind()]], null, 2); inputDirty = true; saveDraft(); };
  $("fit-form").onsubmit = guarded(async event => {
    event.preventDefault();
    if ($("fit-input-method").value === "paste" && $("fit-input").value.trim() || $("fit-input-method").value === "manual" && manualRoles[dataKind()].some(role => String(manualValues[role] ?? "").trim())) {
      if (!await parseInput()) return;
    }
    const invalid = $("fit-table").querySelector("input:invalid");
    if (invalid) { invalid.reportValidity(); throw new Error("Correct the invalid observation field before fitting."); }
    const request = { ...controlsRequest(), observations };
    const queued = await api("/api/fitting", request); await followJob(queued.job_id);
  });
  $("fit-prefill").onclick = guarded(async () => { const queued = await api("/api/fitting/prefill", controlsRequest({ includePsat: false })); await followJob(queued.job_id, "fit_prefill"); });
  $("fit-cancel").onclick = guarded(async () => { if (activeJob) await api(`/api/jobs/${encodeURIComponent(activeJob)}/cancel`, {}); });
  $("fit-project").onchange = guarded(async () => { await selectDefinition(selectedProject($("fit-project"), definitionProject)); });
  $("fit-export-project").onchange = () => { exportProject = selectedProject($("fit-export-project"), exportProject); exportMapping(); saveDraft(); };
  $("fit-import").onclick = () => $("fit-pfd-file").click(); $("fit-export-import").onclick = () => $("fit-export-file").click();
  $("fit-pfd-file").onchange = guarded(async () => { if ($("fit-pfd-file").files[0]) await importPfd($("fit-pfd-file").files[0], "definition"); });
  $("fit-export-file").onchange = guarded(async () => { if ($("fit-export-file").files[0]) await importPfd($("fit-export-file").files[0], "export"); });
  $("fit-copy").onclick = guarded(async () => { await navigator.clipboard.writeText(result.entry); toast("Copied PFD entries."); });
  $("fit-download").onclick = () => download(result.pfd_text, "fitted-mixture.pfd");
  $("fit-report").onclick = () => download(JSON.stringify(result, null, 2), "fit-report.json", "application/json");
  $("fit-merged-download").onclick = guarded(async () => { const exported = await exportMerged(); download(exported.pfd_text, exportProject.filename || "fitted-process.pfd"); });
  $("fit-apply").onclick = guarded(async () => {
    if (!exportProject?.id) throw new Error("Choose a saved laboratory to apply the fit, or download the updated imported PFD.");
    const target = structuredClone(exportProject);
    const latest = readLocal("pfdsim.laboratories.v1", {})[target.id];
    if (latest?.text !== target.text || JSON.stringify(latest?.pfd) !== JSON.stringify(target.pfd)) throw new Error("This laboratory changed in another tab. Select it again before applying the fit.");
    const exported = await exportMerged();
    const library = readLocal("pfdsim.laboratories.v1", {}), current = library[target.id];
    if (current?.text !== target.text || JSON.stringify(current?.pfd) !== JSON.stringify(target.pfd)) throw new Error("This laboratory changed while exporting. Select it again before applying the fit.");
    const saved = { ...current, pfd: exported.pfd, text: exported.pfd_text, updated: Date.now(), pending: false, job: null, lastJob: null };
    library[saved.id] = saved;
    if (!writeLocal("pfdsim.laboratories.v1", library)) throw new Error("Browser storage is full. Download the updated PFD instead.");
    queueCloud(saved); if (exportProject?.id === target.id) exportProject = saved; saveDraft(); toast("Fit applied to the laboratory. Reopen it in Workspace to load the updated definitions.");
  });
  $("fit-submit").onclick = guarded(async () => {
    const source = currentSource();
    const response = await api("/api/fitting/submit", { job_id: jobId, result, source });
    $("fit-submission-status").textContent = `Submitted for review · ${response.submission.id}`;
  });
  $("fit-admin-direct").onclick = guarded(async () => { await adminPublication({ result, source: currentSource(), notes: "Direct root administrator publication" }, "/api/fitting/admin/publish", "fit-submission-status"); });
  const account = await getSession();
  $("fit-publishing-link").hidden = !(account.user?.is_admin && account.user.username.toLowerCase() === "root");
  $("fit-admin-direct").hidden = !account.user?.is_admin;
  await initializeLibrary(); libraryOptions();
  if (saved?.activeJob) {
    const response = await api(`/api/jobs/${encodeURIComponent(saved.activeJob)}`);
    await followJob(saved.activeJob, response.job.kind);
  }
  const reviewId = new URLSearchParams(location.search).get("review");
  if (reviewId && account.user?.is_admin) {
    const { submission } = await api(`/api/fitting/admin/submissions/${encodeURIComponent(reviewId)}`);
    if (await sessionLibrary.openReviewedFit(fittingSessionFromResult(submission.result, submission.source))) {
      const url = new URL(location.href); url.searchParams.delete("review"); history.replaceState(null, "", url);
      $("fit-results").scrollIntoView({block:"start"});
    }
  }
}
guarded(initialize)();
