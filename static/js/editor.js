import {
  $,
  api,
  clone,
  element,
  svgElement,
  toast,
  modal,
  confirmAction,
  download,
  settings,
  applySettings,
  readLocal,
  writeLocal,
  pollJob,
  guarded,
} from "./common.js";
import { equipmentSVG } from "./equipment.js";
import { tabbedForm, labelFor } from "./fields.js";
import { initializeLibrary, queueCloud } from "./persistence.js";
import { renderSimulationData, percentage } from "./results.js";
import { parameterControls } from "./unit-controls.js";
import {
  componentLabels,
  reactionEditor,
  interactionEditor,
  particleEditor,
  correlationEditor,
  modelOptionEditor,
  thermoOptionEditor,
  compositionEditor,
  streamSelectionEditor,
} from "./configuration-editors.js";

const state = {
  pfd: null,
  text: "",
  pending: false,
  id: null,
  filename: "process.pfd",
  selection: null,
  view: "diagram",
  busy: false,
  undo: [],
  redo: [],
  job: null,
  result: null,
  resultText: null,
  pan: { x: 30, y: 30 },
  zoom: 1,
  connecting: null,
  lastJob: null,
  layout: null,
};
let config, templates, drag, inspectorForm, inspectorOriginal;
const source = $("code-editor");
const categoryNames = {
  mixer: "Mixing & routing",
  splitter: "Mixing & routing",
  pump: "Pressure & transport",
  compressor: "Pressure & transport",
  expander: "Pressure & transport",
  valve: "Pressure & transport",
  pipe: "Pressure & transport",
  heater: "Heat transfer",
  cooler: "Heat transfer",
  heat_exchanger: "Heat transfer",
  reactor: "Reaction systems",
  batch_reactor: "Reaction systems",
  crystallizer: "Solids & drying",
  layer_crystallizer: "Solids & drying",
  filter: "Solids & drying",
  dryer: "Solids & drying",
};
const recordTemplates = {
  components: { symbol: "Component", name: "Identifier" },
  params: { name: "", value: "", unit: null },
  properties: { name: "T", value: "25", unit: "C" },
  ports: { id: "", port_type: "inlet" },
  reactions: { equation: "", parameters: {}, reference: null },
  reaction_definitions: { name: "", equation: "", parameters: {} },
  thermo_scopes: { name: "local", method: "UNIFAC", inherit: "global" },
  interaction_parameters: {
    component1: "",
    component2: "",
    model: "NRTL",
    scope: null,
    parameters: {},
  },
  interaction_estimation: {
    model: "NRTL",
    component1: null,
    component2: null,
    scope: null,
    parameters: {},
  },
};
function formOptions(context = state.pfd) {
  const componentSections = {
    identity: {
      label: "Identity",
      keys: [
        "symbol",
        "name",
        "formula",
        "CAS",
        "molecular_weight",
        "smiles",
        "phase_at_STP",
        "phase_behavior",
      ],
    },
    thermo: {
      label: "Thermo",
      keys: [
        "Tc",
        "Pc",
        "Vc",
        "Zc",
        "omega",
        "Tb",
        "Tm",
        "Tt",
        "Pt",
        "Hvap",
        "Hfus",
        "Hf",
        "Gf",
        "S",
        "Hf_liquid",
        "Gf_liquid",
        "S_liquid",
        "Hf_solid",
        "Gf_solid",
        "S_solid",
        "Hcomb",
        "Hcomb_gross",
        "critical_properties_unavailable",
        "uniquac_r",
        "uniquac_q",
        "unifac_groups",
        "vapor_dimerization",
      ],
    },
    solids: {
      label: "Solids",
      keys: [
        "solid_material_form",
        "solid_polymorph",
        "rho_solid",
        "Vm_solid",
        "Cp_solid",
        "particle_diameter",
        "particle_sphericity",
        "particle_size_distribution",
      ],
    },
    advanced: { label: "Advanced", keys: [] },
  };
  const allocated = Object.values(componentSections).flatMap((s) => s.keys);
  componentSections.advanced.keys = Object.keys(
    config.component_defaults,
  ).filter((key) => !allocated.includes(key));
  return {
    templates: recordTemplates,
    defaults: { components: config.component_defaults },
    recordSections: { components: componentSections },
    contextLabels: {
      components: componentLabels,
      thermo_scopes: {
        name: "Scope name",
        method: "Thermodynamic method",
        inherit: "Inherit interaction data from scope",
      },
    },
    contextHelp: {
      components: {
        name: "This is the lookup identifier used by .pfd. You can enter a name, formula, CAS number or SMILES. Other properties are resolved automatically unless overridden.",
        symbol:
          "A short, unique label used in stream compositions, reactions and interaction data.",
        phase_behavior:
          "Use permanent solid for nonparticipating process solids; normal fluid is the default.",
      },
    },
    enumLabels: {
      port_type: Object.fromEntries(
        config.port_types.map((type) => [type, labelFor(type)]),
      ),
      phase_behavior: {
        conventional: "Normal fluid",
        permanent_solid: "Permanent solid",
        conventional_with_solid: "Fluid with explicit solid inventory",
      },
    },
    arrayTypes: { Cp_coeffs: "Number", recycle_tear_streams: "String" },
    recordEditors: {
      reaction_definitions: (record, update) =>
        reactionEditor(record, update, config, {
          components: context.components || [],
          named: true,
        }),
      reactions: (record, update) =>
        reactionEditor(record, update, config, {
          components: context.components || [],
          definitions: context.reaction_definitions || [],
          unitType: context.unitType || "",
        }),
      interaction_parameters: (record, update) =>
        interactionEditor(record, update, config, {
          components: context.components || [],
          scopes: context.thermo_scopes || [],
        }),
      interaction_estimation: (record, update) =>
        interactionEditor(record, update, config, {
          components: context.components || [],
          scopes: context.thermo_scopes || [],
          estimation: true,
        }),
    },
    customEditors: {
      recycle_tear_streams: (value, update) =>
        streamSelectionEditor(
          value,
          update,
          context.streams || state.pfd.streams,
        ),
      composition: (value, update, owner) =>
        compositionEditor(
          value,
          update,
          context.components || [],
          owner.composition_basis || context.composition_basis,
          (basis) => {
            owner.composition_basis = basis;
          },
        ),
      properties: (value, update) =>
        parameterControls(
          value,
          [
            { name: "T", label: "Feed temperature", unit: "C" },
            { name: "P", label: "Stream pressure", unit: "bar" },
            { name: "F", label: "Molar flow", unit: "kmol/h" },
            { name: "F_mass", label: "Mass flow", unit: "kg/h" },
            { name: "vapor_fraction", label: "Vapor fraction (0–1)" },
          ].map((s) => ({
            ...s,
            canonical: s.name.toLowerCase(),
            section: "operation",
            default: null,
            type: "number",
            description:
              "Feed condition or internal-stream initial guess. Choose molar or mass flow, and temperature or vapor fraction.",
          })),
          "operation",
          update,
        ),
      particle_size_distribution: (value, update) =>
        particleEditor(value, update),
      property_correlations: (value, update) =>
        correlationEditor(value, update, config),
      thermo_options: (value, update, owner) =>
        thermoOptionEditor(
          value,
          update,
          owner.thermo_method ||
            context.thermo_method ||
            context.metadata?.thermo_method,
        ),
      recycle_options: (value, update, owner) =>
        modelOptionEditor(
          value,
          update,
          Object.entries(
            config.recycle_defaults[
              owner.recycle_method ||
                context.recycle_method ||
                context.metadata?.recycle_method ||
                "WEGSTEIN"
            ],
          ).map(([name, defaultValue]) => ({
            name,
            label: labelFor(name),
            type: "number",
            default: defaultValue,
          })),
          "Unchecked controls use the selected recycle method’s defaults.",
        ),
    },
    types: {
      components: config.component_types,
      psat_minimum_pressure_bar: "float",
      recycle_trace_tolerance: "float",
      unit: "string",
      reference: "string",
      inherit: "string",
      scope: "string",
      component1: "string",
      component2: "string",
      composition: "dict",
    },
    enums: {
      thermo_method: config.thermo_methods,
      fluid_phase_model: config.fluid_phase_models,
      recycle_method: Object.keys(config.recycle_defaults),
      port_type: config.port_types,
      composition_basis: ["mole", "mass"],
      phase_behavior: config.phase_behaviors,
      solid_material_form: config.solid_material_forms,
    },
    labels: {
      online_lookup: "Allow online property lookup",
      allow_computation: "Allow property computation",
      psat_minimum_pressure_bar: "Minimum saturation pressure (bar)",
      recycle_trace_tolerance: "Trace-component recycle tolerance",
    },
    help: {
      thermo_method:
        "Changing the model resets its additional options to defaults.",
      recycle_method:
        "Changing the method resets its additional options to defaults.",
      thermo_options:
        "Options for the selected model, such as correlation for second-virial methods.",
      recycle_options:
        "Method-specific controls. Choose the method first, then add its supported options.",
      recycle_tear_streams: "Stream IDs used as recycle tears.",
      composition:
        "Component fractions. Values must be numbers; choose a mole or mass basis.",
      particle_size_distributions:
        "Component-specific distribution specifications.",
      x: "Position on the diagram; kept in the browser laboratory.",
      y: "Position on the diagram; kept in the browser laboratory.",
    },
  };
}
function snapshot() {
  return {
    pfd: clone(state.pfd),
    text: state.text,
    pending: state.pending,
    filename: state.filename,
    layout: state.layout,
  };
}
function stash() {
  if (!state.pfd || !state.id) return;
  const library = readLocal("pfdsim.laboratories.v1", {});
  library[state.id] = {
    ...library[state.id],
    ...snapshot(),
    id: state.id,
    updated: Date.now(),
    job: state.job,
    lastJob: state.lastJob,
  };
  const saved =
    writeLocal("pfdsim.laboratories.v1", library) &&
    writeLocal("pfdsim.last.v1", state.id);
  $("save-status").textContent = saved
    ? state.pending
      ? "Text draft saved locally"
      : "Saved locally"
    : "Local save unavailable · export your work";
  queueCloud(library[state.id]);
  return saved;
}
function checkpoint() {
  state.undo.push(snapshot());
  if (state.undo.length > 80) state.undo.shift();
  state.redo = [];
}
function restore(saved) {
  Object.assign(state, clone(saved));
  source.value = state.text;
  state.selection = null;
  state.connecting = null;
  render();
  renderInspector();
  stash();
}
function ensureEditable() {
  if (state.busy) throw new Error("Wait for the current edit to finish.");
  if (state.pending)
    throw new Error(
      "Apply or discard your source edits before changing the diagram.",
    );
}
function busy(value) {
  state.busy = value;
  source.readOnly = value;
  $("workspace-status").textContent = value
    ? "Applying changes…"
    : "Ready to explore";
  $("undo-button").disabled = !state.undo.length || value;
  $("redo-button").disabled = !state.redo.length || value;
}
async function commit(change) {
  ensureEditable();
  const draft = clone(state.pfd);
  change(draft);
  busy(true);
  try {
    const data = await api("/api/serialize", { pfd: draft });
    const drawing = await fetchLayout(data.pfd);
    checkpoint();
    state.pfd = drawing.pfd;
    state.layout = drawing.layout;
    state.text = data.text;
    state.pending = false;
    source.value = state.text;
    render();
    renderInspector();
    stash();
  } finally {
    busy(false);
  }
}
async function applySource() {
  if (state.busy) throw new Error("Wait for the current edit to finish.");
  busy(true);
  try {
    const data = await api("/api/parse", { text: state.text });
    // Keep local positions when text changes leave a unit in place.
    for (const unit of data.pfd.units) {
      const old = state.pfd.units.find((u) => u.id === unit.id);
      if (old) {
        unit.x = old.x;
        unit.y = old.y;
      }
    }
    const drawing = await fetchLayout(data.pfd);
    checkpoint();
    state.pfd = drawing.pfd;
    state.layout = drawing.layout;
    state.pending = false;
    state.selection = null;
    render();
    renderInspector();
    stash();
    toast("Source applied to the flowsheet.");
  } catch (error) {
    $("source-status").textContent = error.message;
    throw error;
  } finally {
    busy(false);
  }
}
function render() {
  $("project-name").textContent =
    state.pfd.metadata.process_name || "Untitled laboratory";
  $("active-method").textContent =
    `${state.pfd.metadata.thermo_method} · ${state.pfd.metadata.fluid_phase_model}`;
  $("component-count").textContent = state.pfd.components.length;
  $("flowsheet-counts").textContent =
    `${state.pfd.units.length} UNITS / ${state.pfd.streams.length} STREAMS`;
  $("empty-workspace").hidden =
    state.pfd.units.length > 0 || state.pfd.streams.length > 0;
  $("source-status").textContent = state.pending
    ? "Unapplied text draft"
    : "Source synchronized";
  $("undo-button").disabled = !state.undo.length || state.busy;
  $("redo-button").disabled = !state.redo.length || state.busy;
  $("result-indicator").textContent =
    state.resultText && state.resultText !== state.text ? "•" : "";
  if (state.view === "results") renderResults();
  renderDiagram();
}
function showView(view) {
  state.view = view;
  for (const name of ["diagram", "code", "results"])
    $(`${name}-panel`).hidden = name !== view;
  document.querySelectorAll("[data-view]").forEach((b) => {
    const active = b.dataset.view === view;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active);
  });
  if (view === "results") renderResults();
}
function route(path, push = true) {
  if (push) history.pushState({}, "", path);
  $("title-screen").hidden = path !== "/";
  $("settings-screen").hidden = path !== "/settings";
  $("workspace").hidden = path !== "/editor";
  if (path === "/settings") renderSettings();
  if (path === "/editor") requestAnimationFrame(renderDiagram);
}
function inspectorDirty() {
  return (
    inspectorForm &&
    JSON.stringify(inspectorForm.value()) !== JSON.stringify(inspectorOriginal)
  );
}
function protectDraft(action) {
  if (inspectorDirty())
    confirmAction(
      "Unapplied inspector changes",
      "Apply your inspector changes first, or discard this draft to continue.",
      () => {
        renderInspector();
        action();
      },
      "Discard draft",
    );
  else action();
}
function select(selection) {
  protectDraft(() => {
    state.selection = selection;
    state.connecting = null;
    if (selection?.type === "unit" && state.zoom < 0.35) {
      const unit = state.pfd.units.find((u) => u.id === selection.id);
      if (unit) {
        const rect = $("flowsheet").getBoundingClientRect(),
          size = unitSize(unit);
        state.zoom = Math.min(
          1.2,
          (rect.width - 100) / (size.width + 160),
          (rect.height - 80) / (size.height + 160),
        );
        state.pan = {
          x: rect.width / 2 - (unit.x + size.width / 2) * state.zoom,
          y: rect.height / 2 - (unit.y + size.height / 2) * state.zoom,
        };
      }
    }
    renderDiagram();
    renderInspector();
  });
}

function inlet(port) {
  return port.port_type === "inlet" || port.port_type.endsWith("_inlet");
}
function unitSize(unit) {
  const layout = state.layout?.units[unit.id];
  return { width: layout?.width || 180, height: layout?.height || 194 };
}
async function fetchLayout(pfd, rearrange = false) {
  return api("/api/layout", { pfd, keep_positions: !rearrange });
}
async function refreshLayout(rearrange = false) {
  const data = await fetchLayout(state.pfd, rearrange);
  state.pfd = data.pfd;
  state.layout = data.layout;
}
function portPosition(unit, port) {
  const mapped = state.layout?.units[unit.id]?.ports[port.id];
  if (mapped) return { x: unit.x + mapped.x, y: unit.y + mapped.y };
  const list = unit.ports.filter((p) => inlet(p) === inlet(port));
  return {
    x: unit.x + (inlet(port) ? 0 : unitSize(unit).width),
    y: unit.y + 42 + ((list.indexOf(port) + 1) * 120) / (list.length + 1),
  };
}
function endpoint(ref, stream, isSource) {
  const unit = state.pfd.units.find((u) => u.id === ref.unit_id);
  if (unit) {
    const port = unit.ports.find((p) => p.id === ref.port_id);
    if (port) return portPosition(unit, port);
    return { x: unit.x + (isSource ? 180 : 0), y: unit.y + 95 };
  }
  const other = isSource ? stream.destination : stream.source;
  const target = state.pfd.units.find((u) => u.id === other.unit_id);
  const stored = state.layout?.streams[stream.id];
  if (stored) {
    const point = isSource ? stored.points[0] : stored.points.at(-1);
    const planned = state.layout.units[other.unit_id];
    return {
      x: point[0] + (target && planned ? target.x - planned.x : 0),
      y: point[1] + (target && planned ? target.y - planned.y : 0),
    };
  }
  const related = state.pfd.streams.filter((s) =>
    isSource
      ? s.source.is_feed && s.destination.unit_id === other.unit_id
      : s.destination.is_product && s.source.unit_id === other.unit_id,
  );
  const index = related.indexOf(stream);
  return {
    x: (target?.x || 150) + (isSource ? -100 : 280),
    y: (target?.y || 100) + 55 + Math.max(0, index) * 50,
  };
}
function renderDiagram() {
  if (!state.pfd) return;
  $("world").setAttribute(
    "transform",
    `translate(${state.pan.x} ${state.pan.y}) scale(${state.zoom})`,
  );
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
  $("streams-layer").replaceChildren();
  $("equipment-layer").replaceChildren();
  for (const stream of state.pfd.streams) {
    const a = endpoint(stream.source, stream, true),
      b = endpoint(stream.destination, stream, false);
    const mid = (a.x + b.x) / 2;
    const stored = state.layout?.streams[stream.id];
    const matches =
      stored &&
      Math.hypot(stored.points[0][0] - a.x, stored.points[0][1] - a.y) < 0.1 &&
      Math.hypot(stored.points.at(-1)[0] - b.x, stored.points.at(-1)[1] - b.y) <
        0.1;
    const d = matches
      ? "M" + stored.points.map((p) => p.join(" ")).join(" L")
      : `M${a.x} ${a.y} C${a.x + Math.max(40, Math.abs(b.x - a.x) * 0.4)} ${a.y},${b.x - Math.max(40, Math.abs(b.x - a.x) * 0.4)} ${b.y},${b.x} ${b.y}`;
    const group = svgElement("g", {
      class: `stream-group${state.selection?.type === "stream" && state.selection.id === stream.id ? " selected" : ""}`,
      "data-stream": stream.id,
      tabindex: 0,
      role: "button",
      "aria-label": `Stream ${stream.id}`,
    });
    group.append(
      svgElement("path", { class: "stream-hit", d }),
      svgElement("path", { class: "stream-path", d }),
    );
    const label = svgElement("text", {
      class: "stream-label",
      x: matches ? stored.label.x : mid,
      y: matches ? stored.label.y : (a.y + b.y) / 2 - 9,
      "text-anchor": "middle",
    });
    label.textContent = stream.id;
    group.append(label);
    for (const [ref, point] of [
      [stream.source, a],
      [stream.destination, b],
    ])
      if (ref.is_feed || ref.is_product) {
        const boundary = stored?.[ref.is_feed ? "feed" : "product"];
        group.append(
          svgElement("rect", {
            class: "boundary-node",
            x: boundary?.x ?? point.x - 30,
            y: boundary?.y ?? point.y - 14,
            width: boundary?.width || 60,
            height: boundary?.height || 28,
            rx: 7,
          }),
        );
        const text = svgElement("text", {
          class: "boundary-label",
          x: boundary ? boundary.x + boundary.width / 2 : point.x,
          y: boundary ? boundary.y + boundary.height / 2 + 3 : point.y + 3,
        });
        text.textContent = ref.is_feed ? "FEED" : "PRODUCT";
        group.append(text);
      }
    group.onpointerdown = (e) => {
      e.stopPropagation();
      select({ type: "stream", id: stream.id });
    };
    group.onkeydown = (e) => {
      if (e.key === "Enter") select({ type: "stream", id: stream.id });
    };
    $("streams-layer").append(group);
  }
  for (const unit of state.pfd.units) {
    const size = unitSize(unit);
    const group = svgElement("g", {
      class: `unit-node${state.selection?.type === "unit" && state.selection.id === unit.id ? " selected" : ""}`,
      transform: `translate(${unit.x} ${unit.y})`,
      "data-unit": unit.id,
      tabindex: 0,
      role: "button",
      "aria-label": `${unit.unit_type} ${unit.id}`,
    });
    group.append(
      svgElement("rect", {
        class: "unit-backdrop",
        width: size.width,
        height: size.height,
        rx: 15,
      }),
    );
    // Only our own registered artwork enters innerHTML; names are text nodes.
    const artwork = svgElement("g", {
      transform: `translate(${(size.width - 132) / 2} 40) scale(.82)`,
    });
    artwork.innerHTML = equipmentSVG(unit.unit_type);
    group.append(artwork);
    const title = svgElement("text", {
      class: "unit-name",
      x: size.width / 2,
      y: 20,
    });
    title.textContent = unit.id;
    const subtitle = svgElement("text", {
      class: "unit-type",
      x: size.width / 2,
      y: 34,
    });
    subtitle.textContent = unit.unit_type;
    group.append(title, subtitle);
    for (const port of unit.ports) {
      const point = portPosition(unit, port);
      const incoming = inlet(port);
      const left = point.x - unit.x < size.width / 2;
      const circle = svgElement("circle", {
        class: `port ${incoming ? "inlet" : "outlet"}${state.connecting?.unit_id === unit.id && state.connecting?.port_id === port.id ? " connecting" : ""}`,
        cx: point.x - unit.x,
        cy: point.y - unit.y,
        r: 6,
        tabindex: 0,
        role: "button",
        "aria-label": `${unit.id}.${port.id} ${incoming ? "inlet" : "outlet"}`,
        "data-port": port.id,
      });
      const connect = (e) => {
        e.stopPropagation();
        e.preventDefault();
        protectDraft(() => guarded(() => connectPort(unit, port))());
      };
      circle.onpointerdown = connect;
      circle.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") connect(e);
      };
      const text = svgElement("text", {
        class: "port-label",
        x: left ? 9 : size.width - 9,
        y: point.y - unit.y + 3,
        "text-anchor": left ? "start" : "end",
      });
      text.textContent = port.id;
      group.append(circle, text);
    }
    group.onpointerdown = (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      if (inspectorDirty()) {
        select({ type: "unit", id: unit.id });
        return;
      }
      state.selection = { type: "unit", id: unit.id };
      renderInspector();
      if (state.pending || state.busy) {
        renderDiagram();
        return;
      }
      const point = worldPoint(e);
      drag = {
        unitId: unit.id,
        start: point,
        x: unit.x,
        y: unit.y,
        before: snapshot(),
        moved: false,
      };
      $("flowsheet").setPointerCapture(e.pointerId);
      renderDiagram();
    };
    group.onkeydown = (e) => {
      if (e.key === "Enter") select({ type: "unit", id: unit.id });
    };
    $("equipment-layer").append(group);
  }
  $("interaction-hint").textContent = state.connecting
    ? "Choose an inlet · Escape to cancel"
    : "Drag equipment · Scroll to zoom · Drag empty space to pan";
}
function worldPoint(e) {
  const rect = $("flowsheet").getBoundingClientRect();
  return {
    x: (e.clientX - rect.left - state.pan.x) / state.zoom,
    y: (e.clientY - rect.top - state.pan.y) / state.zoom,
  };
}
function uniqueId(prefix, items) {
  let n = 1;
  while (items.some((item) => item.id === `${prefix}-${n}`)) n++;
  return `${prefix}-${n}`;
}
async function connectPort(unit, port) {
  ensureEditable();
  if (!state.connecting) {
    if (inlet(port))
      throw new Error(
        "Choose an outlet first. Use Connect stream to create a feed.",
      );
    state.connecting = { unit_id: unit.id, port_id: port.id };
    renderDiagram();
    return;
  }
  if (!inlet(port))
    throw new Error("Choose an inlet to complete this connection.");
  const connection = clone(state.connecting);
  if (
    state.pfd.streams.some(
      (s) =>
        s.destination.unit_id === unit.id && s.destination.port_id === port.id,
    )
  )
    throw new Error(
      "This inlet is already connected. Add another inlet in the Ports tab.",
    );
  await commit((pfd) =>
    pfd.streams.push({
      id: uniqueId("S", pfd.streams),
      source: { ...connection, is_feed: false, is_product: false },
      destination: {
        unit_id: unit.id,
        port_id: port.id,
        is_feed: false,
        is_product: false,
      },
      properties: [],
      composition: null,
      composition_basis: "mole",
      particle_size_distributions: {},
    }),
  );
  state.connecting = null;
  $("connection-layer").replaceChildren();
  renderDiagram();
}
function fit() {
  const units = state.pfd.units;
  if (!units.length) {
    state.pan = { x: 30, y: 30 };
    state.zoom = 1;
    renderDiagram();
    return;
  }
  const rect = $("flowsheet").getBoundingClientRect();
  const bounds = state.layout?.bounds;
  const minX = bounds?.x ?? Math.min(...units.map((u) => u.x)) - 110,
    minY = bounds?.y ?? Math.min(...units.map((u) => u.y)) - 30,
    maxX = bounds
      ? bounds.x + bounds.width
      : Math.max(...units.map((u) => u.x + unitSize(u).width)) + 110,
    maxY = bounds
      ? bounds.y + bounds.height
      : Math.max(...units.map((u) => u.y + unitSize(u).height)) + 30;
  state.zoom = Math.max(
    0.03,
    Math.min(
      1.4,
      (rect.width - 60) / (maxX - minX),
      (rect.height - 70) / (maxY - minY),
    ),
  );
  state.pan = {
    x: (rect.width - (maxX - minX) * state.zoom) / 2 - minX * state.zoom,
    y: (rect.height - (maxY - minY) * state.zoom) / 2 - minY * state.zoom,
  };
  renderDiagram();
}
function zoom(factor, clientX, clientY) {
  const rect = $("flowsheet").getBoundingClientRect();
  const x = clientX ?? rect.width / 2,
    y = clientY ?? rect.height / 2;
  const next = Math.max(0.03, Math.min(3, state.zoom * factor));
  state.pan = {
    x: x - ((x - state.pan.x) * next) / state.zoom,
    y: y - ((y - state.pan.y) * next) / state.zoom,
  };
  state.zoom = next;
  renderDiagram();
}
$("flowsheet").onpointerdown = (e) => {
  if (e.button !== 0) return;
  select(null);
  drag = { pan: true, x: e.clientX, y: e.clientY, start: clone(state.pan) };
  $("flowsheet").setPointerCapture(e.pointerId);
};
$("flowsheet").onpointermove = (e) => {
  if (drag?.pan) {
    state.pan = {
      x: drag.start.x + e.clientX - drag.x,
      y: drag.start.y + e.clientY - drag.y,
    };
    renderDiagram();
  } else if (drag?.unitId) {
    const point = worldPoint(e);
    const unit = state.pfd.units.find((u) => u.id === drag.unitId);
    const snap = (v) => (settings.snap ? Math.round(v / 20) * 20 : v);
    unit.x = snap(drag.x + point.x - drag.start.x);
    unit.y = snap(drag.y + point.y - drag.start.y);
    drag.moved = unit.x !== drag.x || unit.y !== drag.y;
    renderDiagram();
  }
  if (state.connecting) {
    const unit = state.pfd.units.find((u) => u.id === state.connecting.unit_id),
      port = unit.ports.find((p) => p.id === state.connecting.port_id),
      a = portPosition(unit, port),
      b = worldPoint(e);
    $("connection-layer").replaceChildren(
      svgElement("path", {
        class: "preview-connection",
        d: `M${a.x} ${a.y}L${b.x} ${b.y}`,
      }),
    );
  }
};
async function finishDrag() {
  const finished = drag;
  drag = null;
  if (finished?.moved) {
    state.undo.push(finished.before);
    state.redo = [];
    busy(true);
    try {
      await refreshLayout();
      stash();
      render();
      renderInspector();
    } catch (error) {
      state.undo.pop();
      restore(finished.before);
      throw error;
    } finally {
      busy(false);
    }
  }
}
$("flowsheet").onpointerup = guarded(finishDrag);
$("flowsheet").onpointercancel = guarded(finishDrag);
$("flowsheet").onwheel = (e) => {
  e.preventDefault();
  const rect = $("flowsheet").getBoundingClientRect();
  zoom(e.deltaY > 0 ? 0.9 : 1.1, e.clientX - rect.left, e.clientY - rect.top);
};
$("flowsheet").ondragover = (e) => e.preventDefault();
$("flowsheet").ondrop = guarded(async (e) => {
  e.preventDefault();
  const type = e.dataTransfer.getData("pfdsim/unit");
  const point = worldPoint(e);
  if (templates[type])
    protectDraft(() => guarded(() => addUnit(type, point))());
});
async function addUnit(type, point) {
  const rect = $("flowsheet").getBoundingClientRect();
  point ??= {
    x: (rect.width / 2 - state.pan.x) / state.zoom - 90,
    y: (rect.height / 2 - state.pan.y) / state.zoom - 90,
  };
  while (
    state.pfd.units.some((unit) => {
      const size = unitSize(unit);
      return (
        point.x < unit.x + size.width + 12 &&
        point.x + 180 + 12 > unit.x &&
        point.y < unit.y + size.height + 12 &&
        point.y + 194 + 12 > unit.y
      );
    })
  )
    point.y += 240;
  let id;
  await commit((pfd) => {
    id = uniqueId(
      type.replace(/[a-z]/g, "").slice(0, 5) || type.slice(0, 3).toUpperCase(),
      pfd.units,
    );
    pfd.units.push({
      id,
      unit_type: type,
      x: Math.round(point.x / 20) * 20,
      y: Math.round(point.y / 20) * 20,
      ports: clone(templates[type].ports),
      params: [],
      reactions: [],
    });
  });
  state.selection = { type: "unit", id };
  render();
  renderInspector();
}
function renderPalette() {
  const query = $("equipment-search").value.toLowerCase();
  $("equipment-palette").replaceChildren();
  let category = "";
  for (const [type, template] of Object.entries(templates).sort(
    (a, b) =>
      categoryName(a[1]).localeCompare(categoryName(b[1])) ||
      a[0].localeCompare(b[0]),
  )) {
    const label = labelFor(type);
    if (
      !label.toLowerCase().includes(query) &&
      !type.toLowerCase().includes(query)
    )
      continue;
    const next = categoryName(template);
    if (next !== category) {
      $("equipment-palette").append(
        element("div", { class: "palette-category" }, next),
      );
      category = next;
    }
    const button = element("button", {
      class: "palette-item",
      draggable: "true",
      "data-equipment": type,
      "aria-label": `Add ${type}`,
    });
    const icon = element("span");
    icon.innerHTML = equipmentSVG(type);
    button.append(icon, element("span", {}, label));
    button.onclick = guarded(() =>
      protectDraft(() => guarded(() => addUnit(type))()),
    );
    button.ondragstart = (e) => e.dataTransfer.setData("pfdsim/unit", type);
    $("equipment-palette").append(button);
  }
}
function categoryName(template) {
  return categoryNames[template.category] || "Separation & columns";
}

function renderInspector() {
  inspectorForm = null;
  inspectorOriginal = null;
  const selected = state.selection;
  $("inspector-tabs").replaceChildren();
  $("inspector-content").replaceChildren();
  $("delete-selection").hidden = !selected;
  if (!selected) {
    $("inspector-heading").replaceChildren(
      element("h2", {}, "Process overview"),
      element("p", {}, "Select equipment or a stream to configure it."),
    );
    for (const [label, value] of [
      ["Equipment", state.pfd.units.length],
      ["Streams", state.pfd.streams.length],
      ["Components", state.pfd.components.length],
      ["Thermodynamics", state.pfd.metadata.thermo_method],
    ]) {
      const row = element("div", { class: "overview-stat" });
      row.append(element("span", {}, label), element("strong", {}, value));
      $("inspector-content").append(row);
    }
    const note = element(
      "p",
      { class: "field-help" },
      "Connect an outlet to an inlet. Use Process settings for components, thermodynamic scopes, interaction data, reactions, and recycle controls.",
    );
    note.style.marginTop = "20px";
    $("inspector-content").append(note);
    return;
  }
  const items = selected.type === "unit" ? state.pfd.units : state.pfd.streams;
  const item = items.find((i) => i.id === selected.id);
  if (!item) {
    state.selection = null;
    renderInspector();
    return;
  }
  $("inspector-heading").replaceChildren(
    element("h2", {}, item.id),
    element(
      "p",
      {},
      selected.type === "unit" ? labelFor(item.unit_type) : "Material stream",
    ),
  );
  const schema =
    selected.type === "unit" ? templates[item.unit_type].settings : [];
  const schemaNames = new Map(schema.map((s) => [s.name.toLowerCase(), s]));
  const sectionOf = (p) =>
    schemaNames.get(p.name.toLowerCase())?.section || "equipment";
  const formValue =
    selected.type === "unit"
      ? {
          ...clone(item),
          basic_params: item.params.filter((p) => sectionOf(p) === "operation"),
          equipment_params: item.params.filter(
            (p) => sectionOf(p) === "equipment",
          ),
          solver_params: item.params.filter((p) => sectionOf(p) === "solver"),
        }
      : clone(item);
  const portOrigins = new Map();
  if (selected.type === "unit") {
    for (const port of formValue.ports) {
      port._editor_port_key = crypto.randomUUID();
      portOrigins.set(port._editor_port_key, port.id);
    }
  }
  const sections =
    selected.type === "unit"
      ? {
          specifications: {
            label: "Specs",
            keys: ["id", "unit_type", "basic_params"],
            description:
              templates[item.unit_type].specification_notes ||
              "Configure this unit using .pfd parameter names and units.",
          },
          equipment: {
            label: "Equipment & model",
            keys: ["equipment_params"],
            description:
              "Equipment details, property models and specialized physical options.",
          },
          solver: {
            label: "Solver",
            keys: ["solver_params"],
            description:
              "Numerical tolerances, initialization and convergence controls.",
          },
          ...(templates[item.unit_type].supports_reactions
            ? {
                reactions: {
                  label: "Reactions",
                  keys: ["reactions"],
                  description:
                    "Set an equation and kinetic parameters, or reference a named reaction.",
                },
              }
            : {}),
          ports: {
            label: "Ports",
            keys: ["ports"],
            description:
              "Add variable ports and choose their directions. Existing connections follow renamed port IDs.",
          },
          layout: { label: "Layout", keys: ["x", "y"] },
        }
      : {
          specifications: { label: "Specs", keys: ["id", "properties"] },
          composition: {
            label: "Composition",
            keys: ["composition"],
          },
          connections: {
            label: "Connections",
            keys: ["source", "destination"],
          },
          particles: {
            label: "Particles",
            keys: ["particle_size_distributions"],
          },
        };
  if (
    selected.type === "unit" &&
    !templates[item.unit_type].supports_reactions &&
    item.reactions.length
  ) {
    $("inspector-heading").append(
      element(
        "p",
        { class: "field-help" },
        "This unit does not support reactions. Existing reaction definitions are preserved, but cannot be edited in this inspector.",
      ),
    );
  }
  const options = formOptions({
    ...state.pfd,
    unitType: item.unit_type,
    composition_basis: item.composition_basis,
  });
  if (selected.type === "unit" && !item.ports.length) {
    const addPorts = element("button", {}, "Add standard ports");
    addPorts.onclick = guarded(() =>
      protectDraft(() =>
        guarded(async () => {
          await commit((pfd) => {
            pfd.units.find((unit) => unit.id === item.id).ports = clone(
              templates[item.unit_type].ports,
            );
          });
          toast(
            "Standard ports added. Choose an outlet and then an inlet to connect equipment.",
          );
        })(),
      ),
    );
    $("inspector-heading").append(
      element(
        "p",
        { class: "field-help" },
        "This equipment has no ports yet. Add its standard ports to connect streams, or define ports in the Ports tab.",
      ),
      addPorts,
    );
  }
  options.enums.unit_type = Object.keys(templates);
  options.templates = {
    ...options.templates,
    basic_params: recordTemplates.params,
    equipment_params: recordTemplates.params,
    solver_params: recordTemplates.params,
  };
  options.suggestions = {
    basic_params: templates[item.unit_type]?.parameter_suggestions,
    equipment_params: templates[item.unit_type]?.parameter_suggestions,
  };
  if (selected.type === "unit")
    options.customEditors = {
      ...options.customEditors,
      basic_params: (params, update) =>
        parameterControls(params, schema, "operation", update),
      equipment_params: (params, update) =>
        parameterControls(params, schema, "equipment", update),
      solver_params: (params, update) =>
        parameterControls(params, schema, "solver", update),
    };
  inspectorOriginal = clone(formValue);
  inspectorForm = tabbedForm({
    navigation: $("inspector-tabs"),
    content: $("inspector-content"),
    sections,
    value: formValue,
    options,
    onSave: async (draft) => {
      const portRenames = new Map();
      if (selected.type === "unit") {
        draft.params = [
          ...draft.basic_params,
          ...draft.equipment_params,
          ...draft.solver_params,
        ];
        delete draft.basic_params;
        delete draft.equipment_params;
        delete draft.solver_params;
        for (const port of draft.ports) {
          if (portOrigins.has(port._editor_port_key))
            portRenames.set(portOrigins.get(port._editor_port_key), port.id);
          delete port._editor_port_key;
        }
      }
      const previous = clone(item);
      if (!draft.id.trim()) throw new Error("Provide an ID.");
      if (items.some((i) => i.id === draft.id && i.id !== previous.id))
        throw new Error("This ID already exists.");
      await commit((pfd) => {
        const list = selected.type === "unit" ? pfd.units : pfd.streams;
        const index = list.findIndex((i) => i.id === previous.id);
        list[index] = draft;
        if (selected.type === "unit")
          for (const stream of pfd.streams)
            for (const ref of [stream.source, stream.destination])
              if (ref.unit_id === previous.id) {
                ref.unit_id = draft.id;
                if (portRenames.has(ref.port_id))
                  ref.port_id = portRenames.get(ref.port_id);
              }
        if (selected.type === "stream" && draft.id !== previous.id) {
          pfd.metadata.recycle_tear_streams =
            pfd.metadata.recycle_tear_streams.map((id) =>
              id === previous.id ? draft.id : id,
            );
        }
      });
      state.selection = { type: selected.type, id: draft.id };
      render();
      renderInspector();
      toast("Specifications applied.");
    },
  });
}
function processForm(active = "process") {
  ensureEditable();
  const layout = element("div", { class: "dialog-layout" }),
    nav = element("nav", {
      class: "vertical-tabs",
      "aria-label": "Process settings sections",
    }),
    content = element("div", { class: "dialog-form" });
  layout.append(nav, content);
  const value = {
    ...clone(state.pfd.metadata),
    ...Object.fromEntries(
      [
        "components",
        "thermo_scopes",
        "interaction_parameters",
        "interaction_estimation",
        "reaction_definitions",
      ].map((k) => [k, clone(state.pfd[k])]),
    ),
  };
  const sections = {
    process: {
      label: "Process",
      keys: [
        "process_name",
        "version",
        "description",
        "author",
        "date",
        "online_lookup",
        "allow_computation",
      ],
    },
    components: {
      label: "Components",
      keys: ["components"],
      description:
        "Add a component, then expand its optional fields for phase behavior, properties, correlations, groups, solid forms, and particle data.",
    },
    thermo: {
      label: "Thermodynamics",
      keys: [
        "thermo_method",
        "fluid_phase_model",
        "thermo_options",
        "psat_minimum_pressure_bar",
        "thermo_scopes",
      ],
    },
    interactions: {
      label: "Interaction data",
      keys: ["interaction_parameters", "interaction_estimation"],
    },
    recycle: {
      label: "Recycle",
      keys: [
        "recycle_method",
        "recycle_options",
        "recycle_tear_streams",
        "recycle_trace_tolerance",
      ],
    },
    reactions: { label: "Named reactions", keys: ["reaction_definitions"] },
  };
  tabbedForm({
    navigation: nav,
    content,
    sections,
    active,
    value,
    options: (draft) => formOptions(draft),
    onSave: async (draft) => {
      await commit((pfd) => {
        for (const key of Object.keys(pfd.metadata))
          if (key in draft) pfd.metadata[key] = draft[key];
        for (const key of [
          "components",
          "thermo_scopes",
          "interaction_parameters",
          "interaction_estimation",
          "reaction_definitions",
        ])
          pfd[key] = draft[key];
      });
      $("modal").close();
      toast("Process configuration applied.");
    },
  });
  modal("Process configuration", layout);
}
function streamDialog() {
  ensureEditable();
  const box = element("div");
  box.append(
    element(
      "p",
      {},
      "Connect equipment, create a feed, or take a product. Stream conditions and composition can be edited in the inspector.",
    ),
  );
  const endpointOptions = (direction) => {
    const choices = direction === "source" ? ["FEED"] : ["PRODUCT"];
    for (const unit of state.pfd.units)
      for (const port of unit.ports)
        if (inlet(port) === (direction === "destination"))
          choices.push(`${unit.id}.${port.id}`);
    return choices;
  };
  const selectors = {};
  for (const direction of ["source", "destination"]) {
    const label = element("label", { class: "field" });
    label.append(
      element("span", { class: "field-label" }, labelFor(direction)),
    );
    const input = element("select");
    for (const option of endpointOptions(direction))
      input.append(element("option", { value: option }, option));
    selectors[direction] = input;
    label.append(input);
    box.append(label);
  }
  const button = element("button", { class: "primary" }, "Create stream");
  button.onclick = guarded(async () => {
    const decode = (value) => {
      if (value === "FEED" || value === "PRODUCT")
        return {
          unit_id: null,
          port_id: null,
          is_feed: value === "FEED",
          is_product: value === "PRODUCT",
        };
      const index = value.lastIndexOf(".");
      return {
        unit_id: value.slice(0, index),
        port_id: value.slice(index + 1),
        is_feed: false,
        is_product: false,
      };
    };
    const from = decode(selectors.source.value),
      to = decode(selectors.destination.value);
    if (from.is_feed && to.is_product)
      throw new Error("Connect the stream to equipment.");
    if (
      state.pfd.streams.some(
        (s) =>
          to.unit_id &&
          s.destination.unit_id === to.unit_id &&
          s.destination.port_id === to.port_id,
      )
    )
      throw new Error("This inlet already has a stream.");
    let id;
    await commit((pfd) => {
      id = uniqueId(
        from.is_feed ? "FEED" : to.is_product ? "PRODUCT" : "S",
        pfd.streams,
      );
      pfd.streams.push({
        id,
        source: from,
        destination: to,
        properties: [],
        composition: null,
        composition_basis: "mole",
        particle_size_distributions: {},
      });
    });
    $("modal").close();
    select({ type: "stream", id });
  });
  box.append(button);
  modal("Connect a material stream", box);
}
async function examples() {
  const data = await api("/api/examples");
  const box = element("div");
  const input = element("input", {
    placeholder: "Search example processes…",
    type: "search",
    "aria-label": "Search examples",
  });
  input.style.width = "100%";
  const list = element("div", { class: "example-list" });
  const renderList = () => {
    list.replaceChildren();
    for (const ex of data.examples) {
      if (
        !`${ex.name} ${ex.filename} ${ex.thermo_method}`
          .toLowerCase()
          .includes(input.value.toLowerCase())
      )
        continue;
      const button = element("button", { class: "example-item" });
      const text = element("div");
      text.append(
        element("strong", {}, ex.name),
        element("small", {}, ex.filename),
      );
      button.append(
        text,
        element("span", { class: "badge" }, ex.thermo_method),
      );
      button.onclick = guarded(async () => {
        if (state.busy) throw new Error("Wait for the current edit.");
        const example = await api(
          `/api/examples/${encodeURIComponent(ex.filename)}`,
        );
        await loadProject(example);
        $("modal").close();
        route("/editor");
        fit();
      });
      list.append(button);
    }
  };
  input.oninput = renderList;
  renderList();
  box.append(input, list);
  modal("Choose an experiment", box);
}
async function loadProject(data, saveCurrent = true) {
  if (state.busy) throw new Error("Wait for the current edit to finish.");
  if (saveCurrent) stash();
  state.id = data.id || crypto.randomUUID();
  state.undo = [];
  state.redo = [];
  state.result = null;
  state.resultText = null;
  state.lastJob = data.lastJob || null;
  state.layout = data.layout || null;
  state.job = data.job || null;
  state.selection = null;
  Object.assign(state, {
    pfd: data.pfd,
    text: data.text,
    pending: data.pending || false,
    filename: data.filename || "process.pfd",
  });
  source.value = state.text;
  showView("diagram");
  render();
  renderInspector();
  stash();
  renderResults();
  $("run-status").textContent =
    "Run a simulation to explore stream and equipment results.";
  $("run-progress").textContent = "";
  $("run-button").disabled = !!state.job;
  $("cancel-run").hidden = !state.job;
  if (state.job)
    guarded(() => trackRun(state.job.id, state.job.text, state.id))();
  else if (state.lastJob)
    guarded(() => restoreResult(state.lastJob, state.id))();
}
function savedLaboratories() {
  const box = element("div"),
    list = element("div", { class: "example-list" });
  for (const saved of Object.values(
    readLocal("pfdsim.laboratories.v1", {}),
  ).sort((a, b) => b.updated - a.updated)) {
    const button = element("button", { class: "example-item" });
    const label = element("div");
    label.append(
      element(
        "strong",
        {},
        saved.pfd.metadata.process_name || "Untitled laboratory",
      ),
      element(
        "small",
        {},
        `${new Date(saved.updated).toLocaleString()} · ${saved.pfd.units.length} equipment · ${saved.pending ? "text draft" : "saved"}`,
      ),
    );
    button.append(
      label,
      element("span", { class: "badge" }, saved.pfd.metadata.thermo_method),
    );
    button.onclick = guarded(async () => {
      await loadProject(saved);
      $("modal").close();
      route("/editor");
      fit();
    });
    list.append(button);
    if (saved.cloudConflict) {
      const conflict = element(
        "button",
        { class: "example-item" },
        `${saved.pfd.metadata.process_name} · newer account version (local draft kept)`,
      );
      conflict.onclick = guarded(async () => {
        // Archive the local branch before choosing the newer account version.
        stash();
        const library = readLocal("pfdsim.laboratories.v1", {}),
          copy = clone(saved);
        copy.id = crypto.randomUUID();
        delete copy.cloudOwner;
        delete copy.cloudVersion;
        delete copy.cloudSignature;
        delete copy.cloudConflict;
        copy.pfd.metadata.process_name += " · local draft copy";
        library[copy.id] = copy;
        library[saved.id] = saved.cloudConflict;
        writeLocal("pfdsim.laboratories.v1", library);
        await loadProject(saved.cloudConflict, false);
        $("modal").close();
        route("/editor");
        fit();
      });
      list.append(conflict);
    }
  }
  box.append(list);
  modal("Saved laboratories", box);
}
async function newProject() {
  if (state.busy) throw new Error("Wait for the current edit.");
  const pfd = clone(config.empty_pfd);
  for (const key of [
    "thermo_method",
    "fluid_phase_model",
    "online_lookup",
    "allow_computation",
  ])
    pfd.metadata[key] = settings[key];
  const data = await api("/api/serialize", { pfd });
  await loadProject(data);
  route("/editor");
  fit();
}
async function openFile(e) {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (state.busy) throw new Error("Wait for the current edit.");
  const data = new FormData();
  data.append("file", file);
  const result = await api("/api/upload", data);
  await loadProject(result);
  route("/editor");
  fit();
}
function historyStep(undo) {
  // History snapshots include both the applied model and its text draft.
  // Restoring a pending snapshot must still allow its matching redo.
  if (state.busy) throw new Error("Wait for the current edit to finish.");
  const from = undo ? state.undo : state.redo,
    to = undo ? state.redo : state.undo;
  if (!from.length) return;
  to.push(snapshot());
  restore(from.pop());
}
async function validate() {
  const result = await api("/api/validate", { text: state.text });
  const box = element("div");
  box.append(
    element(
      "div",
      { class: `form-notice${result.errors.length ? " error-notice" : ""}` },
      result.errors.length
        ? `${result.errors.length} error(s) require attention.`
        : "No validation errors.",
    ),
  );
  for (const [label, messages] of [
    ["Errors", result.errors],
    ["Warnings", result.warnings],
    ["Suggestions", result.dof_analysis.suggestions],
  ])
    if (messages.length) {
      box.append(element("h3", {}, label));
      const list = element("ul");
      messages.forEach((message) => list.append(element("li", {}, message)));
      box.append(list);
    }
  box.append(
    element(
      "h3",
      {},
      `Degrees of freedom: ${result.dof_analysis.total_dof} · ${result.dof_analysis.overall_status}`,
    ),
  );
  const table = element("table", { class: "data-table" });
  for (const item of [
    ...result.dof_analysis.units,
    ...result.dof_analysis.streams,
  ]) {
    const row = element("tr");
    for (const cell of [item.id, item.dof ?? "—", item.status, item.message])
      row.append(element("td", {}, cell));
    table.append(row);
  }
  box.append(table);
  modal("Flowsheet validation", box);
}
function renderResults() {
  const content = $("result-content");
  content.replaceChildren();
  if (!state.result) {
    $("download-results").disabled = true;
    return;
  }
  const data = state.result;
  $("download-results").disabled = false;
  const metrics = element("div", { class: "metrics" });
  for (const [label, value] of [
    ["Convergence", data.converged ? "Converged" : "Not converged"],
    ["Iterations", data.iterations],
    ["Mass balance error", percentage(data.mass_balance_error)],
    [
      "Energy balance error",
      percentage(data.energy_balance_error),
    ],
  ]) {
    const cell = element("div", { class: "metric" });
    cell.append(element("span", {}, label), element("strong", {}, value));
    metrics.append(cell);
  }
  content.append(metrics);
  if (state.resultText !== state.text)
    content.append(
      element(
        "p",
        { class: "form-notice" },
        "These results belong to an earlier flowsheet revision. Run again to update them.",
      ),
    );
  for (const [label, messages] of [
    ["Errors", data.errors],
    ["Warnings", data.warnings],
  ])
    if (messages?.length) {
      content.append(element("h3", {}, label));
      const list = element("ul");
      messages.forEach((m) => list.append(element("li", {}, m)));
      content.append(list);
    }
  renderSimulationData(content, data);
}
const trackedRuns = new Map();
function savedJobUpdate(laboratoryId, changes) {
  const library = readLocal("pfdsim.laboratories.v1", {});
  if (library[laboratoryId]) {
    Object.assign(library[laboratoryId], changes);
    writeLocal("pfdsim.laboratories.v1", library);
  }
}
async function restoreResult(reference, laboratoryId) {
  let job;
  try {
    ({ job } = await api(`/api/jobs/${reference.id}`));
  } catch (error) {
    if (error.status === 404 && state.id === laboratoryId) {
      state.lastJob = null;
      stash();
      return;
    }
    throw error;
  }
  if (state.id !== laboratoryId) return;
  if (job.status === "completed") {
    state.result = job.output;
    state.resultText = reference.text;
    $("run-status").textContent = job.output.converged
      ? "Experiment complete · converged"
      : "Experiment complete · did not converge";
    renderResults();
    render();
  }
}
async function trackRun(identifier, text, laboratoryId = state.id) {
  if (state.id === laboratoryId) {
    state.job = { id: identifier, text };
    stash();
    $("run-button").disabled = true;
    $("cancel-run").hidden = false;
    $("download-results").disabled = true;
  }
  if (trackedRuns.has(identifier)) return trackedRuns.get(identifier);
  const promise = (async () => {
    try {
      const job = await pollJob(identifier, (job) => {
        if (state.id !== laboratoryId) return;
        $("run-status").textContent =
          job.status === "running"
            ? "Calculating your flowsheet…"
            : job.status === "queued"
              ? "Queued for its calculation worker…"
              : job.error || job.status;
        $("run-progress").textContent = job.progress.join("\n");
      });
      const changes = {
        job: null,
        ...(job.status === "completed"
          ? { lastJob: { id: identifier, text } }
          : {}),
      };
      savedJobUpdate(laboratoryId, changes);
      if (state.id === laboratoryId) {
        state.job = null;
        if (job.status === "completed") {
          state.result = job.output;
          state.resultText = text;
          state.lastJob = changes.lastJob;
          $("run-status").textContent = job.output.converged
            ? "Experiment complete · converged"
            : "Experiment complete · did not converge";
          renderResults();
          render();
          toast("Simulation complete.");
        } else if (job.status === "failed") toast(job.error, true);
        stash();
      }
    } catch (error) {
      if (error.status === 404) {
        savedJobUpdate(laboratoryId, { job: null });
        if (state.id === laboratoryId) {
          state.job = null;
          stash();
          $("run-status").textContent =
            "This run is unavailable or expired. You can start a new simulation.";
        }
        return;
      }
      if (state.id === laboratoryId) {
        $("run-status").textContent =
          `Connection interrupted. The run may still be active. ${error.message}`;
        const button = element("button", {}, "Reconnect to run");
        button.onclick = guarded(() =>
          trackRun(identifier, text, laboratoryId),
        );
        $("run-status").append(document.createElement("br"), button);
      }
      throw error;
    } finally {
      trackedRuns.delete(identifier);
      if (state.id === laboratoryId) {
        $("run-button").disabled = !!state.job;
        $("cancel-run").hidden = !state.job;
      }
    }
  })();
  trackedRuns.set(identifier, promise);
  return promise;
}
let runSubmitting = false;
async function run() {
  if (state.job || runSubmitting)
    throw new Error("A simulation is already active.");
  runSubmitting = true;
  $("run-button").disabled = true;
  const laboratoryId = state.id;
  try {
    if (state.pending) await applySource();
    const text = state.text;
    const data = await api("/api/simulate", {
      text,
      max_iterations: settings.max_iterations,
      tolerance: settings.tolerance,
    });
    runSubmitting = false;
    savedJobUpdate(laboratoryId, { job: { id: data.job_id, text } });
    if (state.id === laboratoryId) {
      state.result = null;
      showView("results");
      $("result-content").replaceChildren();
    }
    await trackRun(data.job_id, text, laboratoryId);
  } finally {
    runSubmitting = false;
    $("run-button").disabled = !!state.job;
  }
}
function renderSettings(active = "appearance") {
  const sections = {
    appearance: {
      label: "Appearance",
      keys: ["theme", "grid", "snap", "motion"],
    },
    simulation: { label: "Simulation", keys: ["max_iterations", "tolerance"] },
    defaults: {
      label: "New laboratories",
      keys: [
        "thermo_method",
        "fluid_phase_model",
        "online_lookup",
        "allow_computation",
      ],
    },
  };
  tabbedForm({
    navigation: $("settings-tabs"),
    content: $("settings-content"),
    sections,
    active,
    value: settings,
    options: {
      enums: {
        theme: ["dark", "light"],
        thermo_method: config.thermo_methods,
        fluid_phase_model: config.fluid_phase_models,
      },
      labels: {
        grid: "Show the diagram grid",
        snap: "Snap equipment to grid",
        motion: "Interface motion",
        max_iterations: "Maximum recycle iterations",
        tolerance: "Convergence tolerance",
        online_lookup: "Enable online property lookup",
        allow_computation: "Enable property computation",
      },
    },
    buttonLabel: "Save settings",
    onSave: async (draft) => {
      if (
        !Number.isInteger(draft.max_iterations) ||
        draft.max_iterations < 1 ||
        draft.max_iterations > 10000
      )
        throw new Error(
          "Maximum iterations must be an integer between 1 and 10000.",
        );
      if (
        !Number.isFinite(draft.tolerance) ||
        draft.tolerance < 1e-12 ||
        draft.tolerance > 0.1
      )
        throw new Error("Tolerance must be between 1e-12 and 0.1.");
      Object.assign(settings, draft);
      if (!writeLocal("pfdsim.settings.v1", settings))
        throw new Error("This browser could not store settings.");
      applySettings();
      toast("Settings saved.");
    },
  });
}
function wire() {
  document
    .querySelectorAll("[data-view]")
    .forEach(
      (button) => (button.onclick = () => showView(button.dataset.view)),
    );
  for (const [id, action] of Object.entries({
    "continue-project": () => route("/editor"),
    "new-project": newProject,
    "title-examples": examples,
    "title-open": () => $("file-input").click(),
    "examples-button": examples,
    "empty-examples": examples,
    "open-button": () => $("file-input").click(),
    "project-button": () => processForm(),
    "components-button": () => processForm("components"),
    "add-stream-button": streamDialog,
    "validate-button": validate,
    "apply-source": applySource,
    "undo-button": () => historyStep(true),
    "redo-button": () => historyStep(false),
    "run-button": run,
    "fit-button": fit,
    "layout-button": async () => {
      ensureEditable();
      checkpoint();
      busy(true);
      try {
        await refreshLayout(true);
        render();
        renderInspector();
        stash();
        fit();
      } finally {
        busy(false);
      }
    },
    "zoom-in": () => zoom(1.2),
    "zoom-out": () => zoom(1 / 1.2),
    "settings-back": () => route("/editor"),
  }))
    $(id).onclick = guarded(() => protectDraft(() => guarded(action)()));
  $("save-button").onclick = () => download(state.text, state.filename);
  $("title-library").onclick = () => protectDraft(savedLaboratories);
  $("download-results").onclick = () => {
    if (state.result)
      download(
        state.result.pfr_content,
        state.filename.replace(/\.pfd$/i, "") + ".pfr",
      );
  };
  $("cancel-run").onclick = guarded(async () => {
    if (state.job) await api(`/api/jobs/${state.job.id}/cancel`, {});
  });
  $("file-input").onchange = guarded(openFile);
  $("equipment-search").oninput = renderPalette;
  $("discard-source").onclick = () => {
    if (!state.pending) return;
    confirmAction(
      "Discard source edits",
      "Replace the text draft with the last applied flowsheet?",
      guarded(async () => {
        const data = await api("/api/serialize", { pfd: state.pfd });
        state.text = data.text;
        state.pending = false;
        source.value = state.text;
        render();
        stash();
      }),
      "Discard text",
    );
  };
  $("delete-selection").onclick = () => {
    const selected = clone(state.selection);
    confirmAction(
      "Delete selection",
      selected.type === "unit"
        ? "Delete this equipment and its connected streams?"
        : "Delete this material stream?",
      guarded(async () => {
        await commit((pfd) => {
          const deleted =
            selected.type === "unit"
              ? pfd.streams
                  .filter(
                    (s) =>
                      s.source.unit_id === selected.id ||
                      s.destination.unit_id === selected.id,
                  )
                  .map((s) => s.id)
              : [selected.id];
          if (selected.type === "unit")
            pfd.units = pfd.units.filter((u) => u.id !== selected.id);
          pfd.streams = pfd.streams.filter((s) => !deleted.includes(s.id));
          pfd.metadata.recycle_tear_streams =
            pfd.metadata.recycle_tear_streams.filter(
              (id) => !deleted.includes(id),
            );
        });
        state.selection = null;
        render();
        renderInspector();
      }),
      "Delete",
    );
  };
  source.oninput = () => {
    state.text = source.value;
    state.pending = true;
    $("source-status").textContent = "Unapplied text draft";
    $("result-indicator").textContent =
      state.resultText && state.resultText !== state.text ? "•" : "";
    stash();
  };
  source.onkeydown = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      source.setRangeText(
        "    ",
        source.selectionStart,
        source.selectionEnd,
        "end",
      );
      source.dispatchEvent(new Event("input"));
    }
  };
  document.querySelectorAll(".masthead a").forEach((link) => {
    if (["/", "/editor", "/settings"].includes(link.getAttribute("href")))
      link.onclick = (e) => {
        e.preventDefault();
        protectDraft(() => route(link.getAttribute("href")));
      };
  });
  window.onpopstate = () => route(location.pathname, false);
  window.addEventListener("beforeunload", (e) => {
    if (state.busy || inspectorDirty() || !stash()) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
  document.onkeydown = (e) => {
    if (e.key === "Escape") {
      state.connecting = null;
      $("connection-layer").replaceChildren();
      renderDiagram();
    }
    if (
      (e.ctrlKey || e.metaKey) &&
      e.key === "Enter" &&
      state.view === "code" &&
      !$("modal").open
    ) {
      e.preventDefault();
      protectDraft(() => guarded(applySource)());
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      download(state.text, state.filename);
    }
    if (
      (e.ctrlKey || e.metaKey) &&
      e.key.toLowerCase() === "z" &&
      !["INPUT", "TEXTAREA"].includes(e.target.tagName)
    ) {
      e.preventDefault();
      protectDraft(() => guarded(() => historyStep(!e.shiftKey))());
    }
  };
  new ResizeObserver(() => {
    if (state.pfd) renderDiagram();
  }).observe($("diagram-panel"));
}
async function init() {
  await initializeLibrary();
  [config, templates] = await Promise.all([
    api("/api/config"),
    api("/api/unit-templates"),
  ]);
  $("equipment-count").textContent = Object.keys(templates).length;
  $("hero-equipment").innerHTML =
    equipmentSVG("HeatExchanger") +
    equipmentSVG("RigorousDistillation") +
    equipmentSVG("Pump");
  const library = readLocal("pfdsim.laboratories.v1", {}),
    last = readLocal("pfdsim.last.v1", null),
    saved = library[last];
  if (saved?.pfd && typeof saved.text === "string") {
    Object.assign(state, saved);
    source.value = state.text;
  } else {
    state.id = crypto.randomUUID();
    state.pfd = clone(config.empty_pfd);
    const data = await api("/api/serialize", { pfd: state.pfd });
    state.text = data.text;
    source.value = data.text;
  }
  wire();
  renderPalette();
  render();
  renderInspector();
  route(location.pathname, false);
  stash();
  if (state.job) guarded(() => trackRun(state.job.id, state.job.text))();
  else if (state.lastJob)
    guarded(() => restoreResult(state.lastJob, state.id))();
}
init().catch((error) => {
  toast(`Could not initialize the laboratory: ${error.message}`, true);
  $("connection-status").textContent = "OFFLINE";
});
