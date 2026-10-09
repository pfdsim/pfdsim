import { $, api, element, guarded, pollJob, download, readLocal, writeLocal } from "./common.js";
import { labeled } from "./fitting-ui.js";
import { adminPublication, refreshAdmin } from "./fitting-admin.js";
import { renderObjectivePlots } from "./fitting-plots.js";

const draftKey = "pfdsim.parameter-publishing.v1";
const controls = ["comp1", "comp2", "model", "basis", "unit", "alpha", "tref", "tmin", "tmax", "extrapolation", "source", "url", "doi", "notes", "fit-method", "statistics", "preview-kind", "preview-temperature", "preview-points"];
let activePreview = null, previewOutput = null, previewSignature = null;
let valueDrafts = {};

function valueKey() { return `${$("publish-model").value}:${$("publish-basis").value}`; }
function values() {
  return Object.fromEntries(Array.from($("publish-values").querySelectorAll("input"), input => [input.dataset.key, input.value]));
}
function parameters() {
  const optional = id => $(id).value.trim() ? Number($(id).value) : null;
  return {components: [$("publish-comp1").value, $("publish-comp2").value], model: $("publish-model").value,
    basis: $("publish-basis").value, unit: $("publish-unit").value, values: values(), alpha: $("publish-model").value === "NRTL" ? Number($("publish-alpha").value) : .3,
    T_ref_K: $("publish-basis").value === "law" ? Number($("publish-tref").value) : 298.15, Tmin_K: optional("publish-tmin"), Tmax_K: optional("publish-tmax"),
    extrapolation: $("publish-extrapolation").value, fit_method: $("publish-fit-method").value, statistics: $("publish-statistics").value};
}
function signature() {
  return JSON.stringify({parameters: parameters(), kind: $("publish-preview-kind").value, T_K: Number($("publish-preview-temperature").value), n_points: Number($("publish-preview-points").value)});
}
function saveDraft() {
  valueDrafts[valueKey()] = values();
  writeLocal(draftKey, {controls: Object.fromEntries(controls.map(name => [name, $(`publish-${name}`).value])), valueDrafts});
  if (previewOutput && signature() !== previewSignature) $("publish-preview-status").textContent = "Inputs changed since this preview. Preview again to check the current parameters.";
}
function renderValues() {
  const model = $("publish-model").value, basis = $("publish-basis").value;
  const nrtl = model === "NRTL", law = basis === "law";
  $("publish-unit-field").hidden = basis !== "energy";
  $("publish-alpha-field").hidden = !nrtl;
  $("publish-alpha").disabled = !nrtl;
  $("publish-tref-field").hidden = !law;
  $("publish-tref").disabled = !law;
  const equations = {
    energy: nrtl ? "τᵢⱼ = aᵢⱼ / (R T), R = 8.314462618 J mol⁻¹ K⁻¹. Enter signed interaction energies." : "τᵢⱼ = exp(−aᵢⱼ / (R T)), R = 8.314462618 J mol⁻¹ K⁻¹. Enter signed interaction energies.",
    tau: nrtl ? "Enter dimensionless, constant τ₁₂ and τ₂₁ (signed values allowed)." : "Enter dimensionless, constant τ₁₂ and τ₂₁ > 0. PFDSim stores ln(τ) coefficients.",
    law: nrtl ? "τᵢⱼ = c + d/T + e h(T) + f T + g T². h(T) = (Tref−T)/T + ln(T/Tref)." : "ln(τᵢⱼ) = a + b/T + c h(T) + d T + e T². h(T) = (Tref−T)/T + ln(T/Tref).",
  };
  $("publish-equation").textContent = equations[basis];
  const fields = nrtl ? ["c", "d", "e", "f", "g"] : ["a", "b", "c", "d", "e"];
  const units = ["dimensionless", "K", "dimensionless", "K⁻¹", "K⁻²"];
  const draft = valueDrafts[valueKey()] || {};
  $("publish-values").replaceChildren();
  for (const field of law ? fields : [null]) for (const direction of ["12", "21"]) {
    const key = field ? `${direction}.${field}` : direction;
    const label = field ? `${direction} · ${field} (${units[fields.indexOf(field)]})` : `${basis === "energy" ? "a" : "τ"}${direction}`;
    const input = element("input", {id: `publish-value-${key.replace(".", "-")}`, type: "number", step: "any", required: "", "data-key": key});
    input.value = draft[key] ?? (basis === "tau" && !nrtl ? 1 : 0);
    input.addEventListener("input", saveDraft);
    $("publish-values").append(labeled(label, input));
  }
}
function validateEntry() {
  if (!$("publish-form").reportValidity()) throw new Error("Complete the parameter fields with valid values.");
}
async function preview() {
  validateEntry();
  for (const id of ["publish-preview-temperature", "publish-preview-points"]) if (!$(id).reportValidity()) return;
  const input = JSON.parse(signature());
  const requestedSignature = JSON.stringify(input);
  $("publish-preview").disabled = true;
  $("publish-preview-status").textContent = "Queuing preview…";
  try {
    const queued = await api("/api/fitting/admin/preview", input);
    activePreview = queued.job_id;
    $("publish-preview-cancel").disabled = false;
    const job = await pollJob(activePreview, updated => $("publish-preview-status").textContent = updated.progress.at(-1) || updated.status);
    if (job.status !== "completed") throw new Error(job.error || `Preview ${job.status}`);
    previewOutput = job.output;
    previewSignature = requestedSignature;
    renderObjectivePlots($("publish-preview-plots"), previewOutput.plots);
    $("publish-preview-note").textContent = previewOutput.note;
    $("publish-preview-warnings").textContent = (previewOutput.result.warnings || []).join("\n");
    $("publish-preview-parameters").textContent = JSON.stringify({components: previewOutput.result.component_names, parameters: previewOutput.result.parameters, reported_fit: previewOutput.result.reported_fit}, null, 2) + "\n\n" + previewOutput.result.pfd_text;
    $("publish-preview-download").disabled = false;
    $("publish-preview-status").textContent = signature() === previewSignature ? "Preview complete. Check signs, units and directions against the source before publishing." : "Inputs changed during this calculation. The displayed preview uses the earlier inputs; preview again.";
  } catch (error) {
    $("publish-preview-status").textContent = `${error.message}${previewOutput ? " The displayed curves are from the previous completed preview." : ""}`;
    throw error;
  } finally {
    activePreview = null;
    $("publish-preview").disabled = false;
    $("publish-preview-cancel").disabled = true;
  }
}
async function initialize() {
  const saved = readLocal(draftKey, null);
  if (saved) {
    for (const [name, value] of Object.entries(saved.controls || {})) if (controls.includes(name)) $(`publish-${name}`).value = value;
    valueDrafts = saved.valueDrafts || {};
  }
  renderValues();
  let previousKey = valueKey();
  for (const name of controls) $(`publish-${name}`).addEventListener("input", () => {
    if (["model", "basis"].includes(name)) {
      valueDrafts[previousKey] = values();
      renderValues();
      previousKey = valueKey();
    }
    saveDraft();
  });
  $("publish-preview").onclick = guarded(preview);
  $("publish-preview-cancel").onclick = guarded(async () => { if (activePreview) await api(`/api/jobs/${encodeURIComponent(activePreview)}/cancel`, {}); });
  $("publish-preview-download").onclick = () => download(previewOutput.result.pfd_text, "manual-parameters-preview.pfd");
  $("publish-form").onsubmit = guarded(async event => {
    event.preventDefault();
    validateEntry();
    if (!$("publish-source").value.trim()) throw new Error("A source citation is required to publish parameters.");
    $("publish-run").disabled = true;
    $("fit-admin-status").textContent = "Validating and queuing direct publication…";
    try {
      await adminPublication({parameters: parameters(), source: {citation: $("publish-source").value.trim(), url: $("publish-url").value.trim(), doi: $("publish-doi").value.trim(), notes: $("publish-notes").value.trim()}, notes: "Direct root publication of sourced manual parameters"});
    } catch (error) {
      $("fit-admin-status").textContent = error.message;
      throw error;
    } finally { $("publish-run").disabled = false; }
  });
  $("fit-admin-refresh").onclick = guarded(refreshAdmin);
  await refreshAdmin();
}
guarded(initialize)();
