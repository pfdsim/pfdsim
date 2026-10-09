import { $, api, element, guarded, toast, modal, pollJob, download, formatNumber } from "./common.js";
import { table, labeled } from "./fitting-ui.js";

export async function adminPublication(data, path = "/api/fitting/admin/publish", statusId = "fit-admin-status") {
  const queued = await api(path, data);
  const job = await pollJob(queued.job_id, updated => $(statusId).textContent = updated.progress.at(-1) || updated.status);
  if (job.status !== "completed") throw new Error(job.error || `Publication ${job.status}`);
  $(statusId).textContent = `${job.output.status}: ${job.output.id}. New simulation packages use the rebuilt table.`;
  if ($("fit-admin-list")) await refreshAdmin();
}
export async function reviewFit(id,loadedSubmission=null) {
  const submission=loadedSubmission||(await api(`/api/fitting/admin/submissions/${encodeURIComponent(id)}`)).submission;
  const box = element("div");
  box.append(element("h3", {}, `${submission.result.component_names.join(" / ")} · ${submission.model}`));
  box.append(element("p", {}, `Status: ${submission.status}. Source: ${submission.source.citation}`));
  if (submission.result.submission_origin === "manual_parameters") {
    box.append(element("p", {}, "Direct parameter entry. Fitting method and statistics below are source-reported; PFDSim did not run a regression for this entry."));
    const reported = submission.result.reported_fit;
    if (reported?.method) box.append(element("p", {}, `Source-reported fitting method: ${reported.method}`));
    if (reported?.statistics) box.append(element("pre", {}, `Source-reported statistics:\n${reported.statistics}`));
    box.append(element("pre", {}, JSON.stringify(submission.result.manual_input, null, 2)));
  }
  const psatRecords = (submission.result.property_provenance || []).filter(record => record.property === "Psat");
  if (psatRecords.length || submission.result.request.psat?.some(Boolean)) {
    box.append(element("h4", {}, "Fitting saturation-pressure basis"));
    box.append(element("p", {}, "Supplied Psat corrections and extended temperature ranges are retained for review and do not block publication. Publication activates liquid activity parameters; simulations use their own saturation-pressure and vapor definitions."));
    if (psatRecords.length) box.append(table(["Component", "Temperature (K)", "Psat (bar)", "Source", "Validity and assessment"], psatRecords.map(record => {
      const index = submission.result.components.indexOf(record.component);
      return [submission.result.component_names[index] || record.component, formatNumber(record.T_K), formatNumber(record.value), record.source, record.notes];
    })));
  }
  const status=element("p",{id:"fit-admin-review-status",role:"status","aria-live":"polite"},submission.status==="approved"?"Approved and saved. This fit remains in the review queue; Publish to runtime activates it in the shared tables.":`Current status: ${submission.status}.`);
  const details = element("pre"); details.textContent = JSON.stringify({ source: submission.source, objective_scores: submission.result.objectives, weights: submission.result.request.weights, parameters: submission.result.parameters, fitting_psat_definitions: submission.result.request.psat, property_provenance: submission.result.property_provenance, warnings: submission.result.warnings, cross_validation: submission.result.cross_validation, review_history: submission.events }, null, 2); box.append(details);
  const notes = element("textarea", { rows: 3, id: "fit-admin-review-notes", placeholder: "Optional review notes" }); notes.value=submission.review_notes||"";box.append(labeled("Review notes (optional)", notes),status);
  const actions = element("div", { class: "form-actions" });
  const artifact = element("button", { type: "button" }, "Download full provenance"); artifact.onclick = () => download(JSON.stringify(submission, null, 2), `activity-fit-${id}.json`, "application/json"); actions.append(artifact);
  const report=element("button",{type:"button"},"Download fit result");report.onclick=()=>download(JSON.stringify(submission.result,null,2),`fit-result-${id}.json`,"application/json");actions.append(report);
  const pfd=element("button",{type:"button"},"Download fitted PFD");pfd.onclick=guarded(async()=>{const exported=await api("/api/fitting/export",{result:submission.result});download(exported.pfd_text,`fitted-mixture-${id}.pfd`);});actions.append(pfd);
  if (submission.result.submission_origin !== "manual_parameters") {
    actions.append(element("a", {href: `/parameter-fitting?review=${encodeURIComponent(id)}`}, "View fit assessment"));
  }
  if (!["published", "publishing"].includes(submission.status)) {
    for (const [action, label] of [["approve", "Approve"], ["reject", "Reject"]]) {
      if(action==="approve"&&submission.status==="approved")continue;
      const button = element("button", { type: "button" }, label);
      button.onclick = async()=>{
        for(const control of actions.querySelectorAll("button"))control.disabled=true;
        status.className="field-help";status.textContent=action==="approve"?"Approving and saving this fit…":"Saving the review decision…";
        try{
          const response=await api("/api/fitting/admin/review",{id,action,notes:notes.value,version:submission.version});
          await reviewFit(id,response.submission);
          $("fit-admin-review-status").scrollIntoView({block:"nearest"});
          $("fit-admin-status").textContent=`${response.submission.result.component_names.join(" / ")} · ${response.submission.status}. Stored in the review queue${action==="approve"?"; publish separately to activate runtime parameters":""}.`;
          try{await refreshAdmin();}catch(error){toast(`The decision was saved, but the queue could not refresh: ${error.message}`,true);}
        }catch(error){status.className="fit-error";status.textContent=error.message;status.scrollIntoView({block:"nearest"});for(const control of actions.querySelectorAll("button"))control.disabled=false;}
      };actions.append(button);
    }
  }
  if (["approved", "published", "publishing"].includes(submission.status)) {
    const publish = element("button", { type: "button", class: "primary" }, submission.status === "published" ? "Rebuild published fit" : submission.status === "publishing" ? "Recover interrupted publication" : "Publish to runtime");
    publish.onclick = guarded(async () => { $("modal").close(); await adminPublication({ id, notes: notes.value }); }); actions.append(publish);
  }
  if (submission.status === "published") {
    const withdraw = element("button", { type: "button" }, "Withdraw and rebuild"); withdraw.onclick = guarded(async () => { if (!notes.value.trim()) throw new Error("Explain the withdrawal in review notes."); $("modal").close(); await adminPublication({ id, notes: notes.value }, "/api/fitting/admin/withdraw"); }); actions.append(withdraw);
  }
  box.append(actions); modal(submission.result.submission_origin === "manual_parameters" ? "Review manual parameters" : "Review activity fit", box);
}
export async function refreshAdmin() {
  const { submissions } = await api("/api/fitting/admin/submissions");
  $("fit-admin-list").replaceChildren(table(["Mixture", "Model", "Objectives", "Status", "Source", ""], submissions.map(item => {
    const button = element("button", { type: "button" }, "Review"); button.onclick = guarded(() => reviewFit(item.id));
    return [item.component_names?.length?item.component_names.join(" / "):`${item.cas1 || "unresolved"} / ${item.cas2 || "unresolved"}`,item.method||item.model,(item.objectives||[]).join(" + "),item.status,item.source.citation,button];
  })));
}
