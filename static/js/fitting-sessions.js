import { $, api, getSession, readLocal, writeLocal, element, modal, guarded, toast, download } from "./common.js";

const libraryKey="pfdsim.fit-sessions.v1",previousKey="pfdsim.fit-session-previous.v1";
const currentKey="pfdsim.fit-session-current.v1";
function validate(document){
  const object=value=>value!==null&&typeof value==="object"&&!Array.isArray(value);
  if(!object(document)||document.type!=="pfdsim_fit_session"||document.schema_version!==1||typeof document.name!=="string"||!document.name.trim()||document.name.length>120)throw new Error("Choose a PFDSim fitting-session JSON file with a name and schema version 1.");
  const state=document.state;
  if(!object(state)||!object(state.controls)||!Array.isArray(state.observations)||!state.observations.every(object))throw new Error("The saved fitting session must contain its settings and observation rows.");
  for(const key of ["weights","manualValues","sigmaValues"])if(state[key]!==undefined&&!object(state[key]))throw new Error(`Invalid session ${key}.`);
  for(const key of ["observationSets","importReports","vaporParameters"])if(state[key]!==undefined&&!Array.isArray(state[key]))throw new Error(`Invalid session ${key}.`);
  if(state.psatValues!==undefined&&(!Array.isArray(state.psatValues)||state.psatValues.length!==2||!state.psatValues.every(object)))throw new Error("The session must contain two component property settings.");
  for(const key of ["definitionProject","exportProject"])if(state[key]&&!object(state[key].pfd))throw new Error(`Invalid session ${key}.`);
  if(state.result&&(!object(state.result)||!Array.isArray(state.result.points)||!Array.isArray(state.result.components)||!object(state.result.parameters)||!object(state.result.objectives)||!object(state.result.optimizer)||!Array.isArray(state.result.warnings)||!Array.isArray(state.result.cross_validation?.folds)))throw new Error("The saved fit result is incomplete.");
  if(!state.observations.every(row=>typeof row.kind==="string"&&["VLE","LLE","VLLE","HE","GAMMA_INF","AZEOTROPE","UCST","LCST"].includes(row.kind)))throw new Error("The saved observations contain an unsupported data type.");
  for(const set of state.observationSets||[])if(!object(set)||!Array.isArray(set.ids))throw new Error("The saved observation sets are incomplete.");
  return structuredClone(document);
}

export function fittingSessionFromResult(result, source = {}) {
  const request = result.request;
  const controls = {
    comp1: result.components[0], comp2: result.components[1],
    model: request.model, vapor: request.vapor, law: request.form,
    alpha: String(request.alpha), "free-alpha": request.fit_alpha,
    r1: "", q1: "", r2: "", q2: "", cv: request.cv.method,
    folds: String(request.cv.folds), tref: String(request.T_ref_K),
    starts: String(request.starts), evaluations: String(request.max_nfev),
    seed: String(request.seed), extrapolation: request.extrapolation,
    online: request.online_lookup, "estimate-properties": request.estimate_properties,
    "hoc-eta-default": request.allow_hoc_eta_default,
    scales: JSON.stringify(request.scales, null, 2),
    initial: JSON.stringify({ ...request.initial, ...result.coefficients }, null, 2),
    bounds: JSON.stringify(request.bounds, null, 2),
    input: "", "input-method": "paste", "data-kind": "AUTO",
    source: source.citation || request.source || "",
    "source-url": source.url || "", "source-doi": source.doi || "",
    "source-notes": source.notes || "",
  };
  if (result.rq) result.rq.forEach((record, index) => {
    controls[`r${index + 1}`] = String(record.r);
    controls[`q${index + 1}`] = String(record.q);
  });
  const psatValues = [0, 1].map(index => {
    const spec = request.psat?.[index];
    return { ...request.component_properties?.[index], ...(spec ? {
      ...spec, ...spec.coefficients,
    } : {}) };
  });
  return validate({
    type: "pfdsim_fit_session", schema_version: 1,
    name: `${result.component_names.join(" / ")} · ${result.method}`.slice(0, 120),
    state: {
      controls, weights: request.weights, observations: request.observations,
      inputDirty: false, result, jobId: null, activeJob: null,
      // Use the fitted binary PFD so imported pure-fluid definitions and fixed
      // vapor corrections survive without requiring the original process scope.
      definitionProject: { id: null, text: result.pfd_text, pfd: result.pfd, filename: "fitted-mixture.pfd" },
      exportProject: null, selectedScope: "global", exportScope: "global", exportMapping: {},
      observationSets: [{ id: "reviewed-fit", name: "Reviewed fit", ids: request.observations.map(row => row.id) }],
      importReports: request.import_report || [], manualValues: {}, sigmaValues: {},
      vaporParameters: request.vapor_parameters || [], psatValues, pendingImport: null,
    },
  });
}

export function fittingSessionLibrary({capture,restore,canOpen}){
  let current=readLocal(currentKey,null),account=null;
  const initialState = capture();
  let openedState = null;
  function currentRecord(record){current=record;if(record)writeLocal(currentKey,Object.fromEntries(["id","name","version","updated","cloudOwner"].map(key=>[key,record[key]])));else writeLocal(currentKey,null);}
  async function entries(){
    try{account=await getSession(true);}catch{account=await getSession();}
    if(current?.cloudOwner&&current.cloudOwner!==account.user?.id)currentRecord(null);
    const local=readLocal(libraryKey,{});
    if(account.user){
      try{
        const remote=await api("/api/fitting/sessions");
        for(const saved of remote.sessions){if(local[saved.id]?.pendingCloud)continue;local[saved.id]={id:saved.id,...saved.document,version:saved.version,updated:saved.updated,cloudOwner:account.user.id};}
        writeLocal(libraryKey,local);
      }catch(error){toast(`Account fitting-session list unavailable; browser copies remain available. ${error.message}`,true);}
    }
    return Object.values(local).filter(item=>!item.cloudOwner||item.cloudOwner===account.user?.id).sort((a,b)=>(b.updated||0)-(a.updated||0));
  }
  function rememberPrevious(){
    if(!writeLocal(previousKey,{type:"pfdsim_fit_session",schema_version:1,name:"Previous unsaved draft",state:capture()}))throw new Error("Could not preserve the current draft. Download it before opening another fit.");
  }
  function openDocument(document,metadata=null){
    if(!canOpen())throw new Error("Wait for the active calculation to finish before opening another fitting session.");
    const checked=validate(document);rememberPrevious();restore(checked.state);currentRecord(metadata);openedState=capture();
    $("modal").close();toast(`Opened ${checked.name}; settings and observations restored without rerunning the fit.`);
  }
  async function saveCurrent(title, newCopy = false) {
    if (!title.trim()) throw new Error("Enter a fitting-session name.");
    const document = validate({type:"pfdsim_fit_session",schema_version:1,name:title.trim(),state:capture()});
    const identity = (await getSession()).user?.id;
    let metadata = newCopy ? null : current;
    if (metadata?.cloudOwner && metadata.cloudOwner !== identity) metadata = null;
    const id = metadata?.id || crypto.randomUUID(), local = readLocal(libraryKey, {});
    if (metadata && local[id] && local[id].updated !== metadata.updated) throw new Error("This saved fit changed in another tab. Save a new copy or reopen it.");
    let version = metadata?.version || 0, updated = Date.now(), pendingCloud = false;
    if (identity) {
      try { const response = await api(`/api/fitting/sessions/${encodeURIComponent(id)}`, {version, document}); version = response.version; updated = response.updated; }
      catch (error) { if (error.status && error.status < 500) throw error; pendingCloud = true; }
    }
    const record = {id, ...document, version, updated, pendingCloud, ...(identity ? {cloudOwner:identity} : {})};
    local[id] = record;
    const cached = writeLocal(libraryKey, local);
    if (!cached && (!identity || pendingCloud)) throw new Error("Browser storage is full and no account copy was saved; download the session JSON instead.");
    currentRecord(record); openedState = document.state;
    return {record, identity, cached};
  }
  function hasUnsavedWork() {
    const state = capture();
    const baseline = openedState || readLocal(libraryKey, {})[current?.id]?.state;
    const encoded = value => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
    if (baseline) return encoded(state) !== encoded(baseline);
    return !!(state.observations.length || state.result || state.definitionProject || state.exportProject || state.pendingImport || state.controls.input?.trim() || encoded(state) !== encoded(initialState));
  }
  async function openReviewedFit(document) {
    if (!canOpen()) throw new Error("Wait for the active calculation to finish before opening another fitting session.");
    const checked = validate(document);
    if (hasUnsavedWork()) {
      const proceed = await new Promise(resolve => {
        const box = element("div"), name = element("input", {id:"fit-switch-name", type:"text", maxlength:120}), status = element("p", {role:"status"});
        name.value = current?.name || "Unsaved fitting session";
        box.append(element("p", {}, "The current page has unsaved fitting work. Save it before opening this reviewed fit, or continue without a named save. The previous draft remains available in Saved fitting sessions."));
        const label = element("label", {class:"field"}); label.append(element("span", {class:"field-label"}, "Current session name"), name); box.append(label);
        const buttons = element("div", {class:"form-actions"});
        const save = element("button", {type:"button", class:"primary"}, "Save and open fit"), skip = element("button", {type:"button"}, "Open without saving"), cancel = element("button", {type:"button"}, "Cancel");
        let settled = false;
        const dialog = $("modal");
        const finish = value => { if (settled) return; settled = true; dialog.removeEventListener("close", dismiss); resolve(value); };
        const dismiss = () => finish(false);
        dialog.addEventListener("close", dismiss);
        save.onclick = async () => {
          for (const button of [save, skip, cancel]) button.disabled = true;
          try { await saveCurrent(name.value); finish(true); }
          catch (error) { status.textContent = error.message; for (const button of [save, skip, cancel]) button.disabled = false; }
        };
        skip.onclick = () => finish(true);
        cancel.onclick = () => { finish(false); dialog.close(); };
        buttons.append(save, skip, cancel); box.append(buttons, status); modal("Save current fitting work?", box);
      });
      if (!proceed) return false;
    }
    openDocument(checked);
    return true;
  }
  async function open(){
    const saved=await entries(),box=element("div"),status=element("p",{role:"status"}),name=element("input",{id:"fit-session-name",type:"text",maxlength:120,placeholder:"Name this fitting session"});
    name.value=current?.name||"";
    box.append(element("p",{},"Save all settings, observations, row flags, uncertainty, import provenance and completed results. Opening a session preserves the previous draft for recovery."));
    const nameLabel=element("label",{class:"field"});nameLabel.append(element("span",{class:"field-label"},"Session name"),name);box.append(nameLabel);
    const select=element("select",{id:"fit-session-list"});select.append(element("option",{value:""},"Choose a saved fit"));
    let visibleRecords=saved;
    function list(records){visibleRecords=records;const chosen=select.value;select.replaceChildren(element("option",{value:""},"Choose a saved fit"));for(const record of records)select.append(element("option",{value:record.id},`${record.name}${record.pendingCloud?" · browser draft":""} · ${new Date(record.updated||Date.now()).toLocaleString()}`));if(readLocal(previousKey,null))select.append(element("option",{value:"previous"},"Previous unsaved draft"));select.value=chosen;}
    list(saved);box.append(select);
    const buttons=element("div",{class:"form-actions"}),save=element("button",{id:"fit-session-save",type:"button",class:"primary"},"Save current fit"),copy=element("button",{id:"fit-session-copy",type:"button"},"Save new copy"),load=element("button",{id:"fit-session-open",type:"button"},"Open"),exportButton=element("button",{id:"fit-session-download",type:"button"},"Download session JSON"),importButton=element("button",{id:"fit-session-import",type:"button"},"Import session JSON"),file=element("input",{id:"fit-session-file",type:"file",accept:".json",hidden:""});
    async function saveFromDialog(newCopy){
      const {record, identity, cached} = await saveCurrent(name.value, newCopy);
      list(Object.values(readLocal(libraryKey, {})).filter(item=>!item.cloudOwner||item.cloudOwner===identity));select.value=record.id;
      status.textContent=`Saved ${record.name} ${identity&&!record.pendingCloud?"to your account and browser":"in this browser"}.${record.pendingCloud?" Account upload is pending; save again when connected.":""}${cached?"":" Browser caching is unavailable; the account copy is saved."}`;
    }
    save.onclick=guarded(()=>saveFromDialog(false));copy.onclick=guarded(()=>saveFromDialog(true));
    load.onclick=guarded(()=>{
      if(!select.value)throw new Error("Choose a saved fitting session.");
      if(select.value==="previous")openDocument(readLocal(previousKey,null));
      else{const record=visibleRecords.find(item=>item.id===select.value);if(!record)throw new Error("This fitting session is no longer available.");openDocument(record,record);}
    });
    exportButton.onclick=guarded(()=>{const document=validate({type:"pfdsim_fit_session",schema_version:1,name:name.value.trim()||current?.name||"Fitting session",state:capture()});download(JSON.stringify(document,null,2),"fitting-session.json","application/json");});
    importButton.onclick=()=>file.click();file.onchange=guarded(async()=>{if(file.files[0])openDocument(JSON.parse(await file.files[0].text()));});
    buttons.append(save,copy,load,exportButton,importButton);box.append(buttons,file,status,element("p",{class:"field-help"},account.user?"Account saves are private and version checked. A stale save keeps the local draft intact; use Save new copy if needed.":"Guest saves stay in this browser. Sign in to save a session to your account, or download JSON for another device."));
    modal("Saved fitting sessions",box);
  }
  return {open, openReviewedFit};
}
