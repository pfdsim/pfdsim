import {
  $,
  api,
  element,
  svgElement,
  toast,
  download,
  readLocal,
  writeLocal,
  pollJob,
  guarded,
  formatNumber,
} from "./common.js";
import { structuredEditor } from "./fields.js";
import { thermoOptionEditor } from "./configuration-editors.js";
import { initializeLibrary } from "./persistence.js";
let activeJob = null,
  data = null,
  config,
  chemicalOptions = [],
  referencesLoaded = false,
  project = null,
  importedProject = null,
  independentOnline = true,
  modelOptions;
const light = document.body.dataset.theme === "light";
const colors = light
  ? ["#146f61", "#8e601a", "#626cc4"]
  : ["#78e6cb", "#f3bf75", "#9ca5f9"];
const neutral = light ? "#516e7c" : "#698696";
const failure = light ? "#ac364b" : "#ff8996";
function plot(result) {
  if (result.chart_type.startsWith("TERNARY")) return plotTernary(result);
  const type = result.chart_type;
  if (type === "VLLE" && !Array.isArray(result.T_bubble))
    throw new Error("This saved binary VLLE result uses the previous phase-fraction view. Generate a new constant-pressure diagram.");
  const temperaturePlot = ["TXY", "VLLE"].includes(type);
  const series =
    type === "XY"
        ? [
            { name: "Equilibrium curve", x: result.x, y: result.y },
            { name: "x = y", x: [0, 1], y: [0, 1] },
          ]
        : [
            {
              name: "Bubble curve · liquid x",
              x: result.x,
              y: temperaturePlot ? result.T_bubble : result.P_bubble,
            },
            {
              name: "Dew curve · vapor y",
              x: result.y,
              y: temperaturePlot ? result.T_dew : result.P_dew,
            },
          ];
  if (result.binodal?.x1.some(v => typeof v === "number")) {
    series.push(
      { name: "Binodal · liquid 1", x: result.binodal.x1, y: result.binodal.T },
      { name: "Binodal · liquid 2", x: result.binodal.x2, y: result.binodal.T },
    );
  }
  const values = series
    .flatMap((s) => s.y)
    .filter((v) => typeof v === "number" && Number.isFinite(v));
  if (!values.length)
    throw new Error("The model returned no finite curve points.");
  const fractionPlot = type === "XY";
  const padding = fractionPlot
    ? 0
    : Math.max((Math.max(...values) - Math.min(...values)) * 0.08, 0.01);
  const ymin = fractionPlot ? 0 : Math.min(...values) - padding,
    ymax = fractionPlot ? 1 : Math.max(...values) + padding;
  const x = (v) => 70 + v * 640,
    y = (v) => 420 - ((v - ymin) / (ymax - ymin)) * 350;
  const svg = svgElement("svg", {
    viewBox: "0 0 780 490",
    role: "img",
    "aria-label": `${type} equilibrium diagram for ${result.comp1_name} and ${result.comp2_name}`,
  });
  for (let i = 0; i <= 5; i++) {
    const xt = i / 5,
      yt = ymin + ((ymax - ymin) * i) / 5;
    svg.append(
      svgElement("path", {
        class: "chart-grid",
        d: `M${x(xt)} 70V420 M70 ${y(yt)}H710`,
      }),
    );
    const xlabel = svgElement("text", {
      class: "chart-text",
      x: x(xt),
      y: 442,
      "text-anchor": "middle",
    });
    xlabel.textContent = xt.toFixed(1);
    const ylabel = svgElement("text", {
      class: "chart-text",
      x: 58,
      y: y(yt) + 4,
      "text-anchor": "end",
    });
    ylabel.textContent = formatNumber(yt, 4);
    svg.append(xlabel, ylabel);
  }
  svg.append(svgElement("path", { class: "chart-axis", d: "M70 70V420H710" }));
  const xLabel = svgElement("text", {
    class: "chart-text",
    x: 390,
    y: 475,
    "text-anchor": "middle",
  });
  xLabel.textContent = `Mole fraction of ${result.comp1_name} (${type === "XY" ? "liquid x" : "liquid x / vapor y"})`;
  const yLabel = svgElement("text", {
    class: "chart-text",
    transform: "translate(16 250) rotate(-90)",
    "text-anchor": "middle",
  });
  yLabel.textContent =
    temperaturePlot
      ? "Temperature (°C)"
      : type === "PXY"
        ? "Pressure (bar)"
        : "Vapor mole fraction y";
  svg.append(xLabel, yLabel);
  if (result.lle_region && type === "TXY") {
    const region = result.lle_region;
    const left = Math.min(region.x1_phase1, region.x1_phase2),
      right = Math.max(region.x1_phase1, region.x1_phase2);
    if (Number.isFinite(left) && Number.isFinite(right))
      svg.append(
        svgElement("rect", {
          x: x(left),
          y: 70,
          width: (right - left) * 640,
          height: 350,
          fill: "#9ca5f9",
          opacity: 0.1,
        }),
      );
  }
  $("chart-legend").replaceChildren();
  series.forEach((s, index) => {
    let d = "",
      previous = false;
    s.x.forEach((vx, i) => {
      const vy = s.y[i];
      if (
        typeof vx !== "number" ||
        typeof vy !== "number" ||
        !Number.isFinite(vx) ||
        !Number.isFinite(vy)
      ) {
        previous = false;
        return;
      }
      d += `${previous ? "L" : "M"}${x(vx)} ${y(vy)} `;
      previous = true;
    });
    svg.append(
      svgElement("path", {
        class: "chart-line",
        d,
        stroke: colors[Math.min(index,2)],
        "stroke-dasharray": type === "XY" && index === 1 ? "5 5" : "",
      }),
    );
    const legend = element("span");
    const dot = element("i", { class: "legend-dot" });
    dot.style.background = colors[Math.min(index,2)];
    legend.append(dot, document.createTextNode(s.name));
    $("chart-legend").append(legend);
  });
  if (result.azeotrope && temperaturePlot) {
    const az = result.azeotrope;
    const marker = svgElement("circle", {
      cx: x(az.x),
      cy: y(az.T),
      r: 5,
      fill: colors[2],
    });
    const title = svgElement("title");
    title.textContent = `Azeotrope: x=${az.x}, T=${az.T} °C`;
    marker.append(title);
    svg.append(marker);
  }
  $("chart-frame").replaceChildren(svg);
  $("chart-title").textContent = `${result.comp1_name} / ${result.comp2_name}`;
  $("chart-information").replaceChildren(
    element(
      "p",
      { class: "form-notice" },
      `${result.method} · ${type === "PXY" ? `${result.temperature_C} °C · ` : ""}${type !== "PXY" ? `${result.pressure_bar} bar · ` : ""}${result.x?.length || result.samples.length} calculated points${type === "VLLE" ? ". Constant pressure; temperature varies. Liquid 1 is the component-1-lean branch. The binodal is sampled below the boiling boundary." : ""}`,
    ),
  );
  for (const [label, value] of [
    ["Azeotrope", result.azeotrope],
    ["Liquid–liquid region", result.lle_region],
  ])
    if (value) {
      const details = element("details", { class: "result-detail" });
      details.open = true;
      details.append(
        element("summary", {}, label),
        element("pre", {}, JSON.stringify(value, null, 2)),
      );
      $("chart-information").append(details);
    }
  showDiagnostics(result);
  $("export-chart").disabled = false;
  $("export-chart-svg").disabled = false;
}
function showDiagnostics(result) {
  const failures =
    result.errors || result.samples?.filter((s) => s.status === "failed") || [];
  if (failures.length) {
    const detail = element("details", { class: "result-detail" });
    detail.append(
      element(
        "summary",
        {},
        `${failures.length} unconverged points · shown as gaps`,
      ),
      element("pre", {}, JSON.stringify(failures, null, 2)),
    );
    $("chart-information").append(detail);
  }
  if (result.samples) {
    const summary = element("details", { class: "result-detail" });
    summary.append(
      element("summary", {}, "Phase compositions and residuals"),
      element("pre", {}, JSON.stringify(result.samples, null, 2)),
    );
    $("chart-information").append(summary);
  }
  if (result.binodal) {
    if (!result.binodal.x1.some(v => typeof v === "number"))
      $("chart-information").append(element("p", { class: "form-notice" },
        "No liquid split was found on the sampled compositions and temperatures. This is not a global stability proof."));
    else {
      const summary = element("details", { class: "result-detail" });
      summary.append(element("summary", {}, "Binodal phase compositions and residuals"),
        element("pre", {}, JSON.stringify(result.binodal.samples, null, 2)));
      $("chart-information").append(summary);
    }
  }
  $("phase-point-detail").replaceChildren();
}
function plotTernary(result) {
  const comps = result.components,
    vertices = [
      [390, 65],
      [95, 440],
      [685, 440],
    ];
  const point = (z) => ({
    x: comps.reduce((s, c, i) => s + (z[c] || 0) * vertices[i][0], 0),
    y: comps.reduce((s, c, i) => s + (z[c] || 0) * vertices[i][1], 0),
  });
  const svg = svgElement("svg", {
    viewBox: "0 0 780 510",
    role: "img",
    "aria-label": `${result.chart_type} ternary diagram for ${comps.map((c, i) => result[`comp${i + 1}_name`]).join(", ")}`,
  });
  for (let component = 0; component < 3; component++)
    for (let tick = 1; tick < 10; tick++) {
      const fraction = tick / 10;
      const a = {
          [comps[component]]: fraction,
          [comps[(component + 1) % 3]]: 1 - fraction,
        },
        b = {
          [comps[component]]: fraction,
          [comps[(component + 2) % 3]]: 1 - fraction,
        };
      const pa = point(a),
        pb = point(b);
      svg.append(
        svgElement("path", {
          class: "chart-grid",
          d: `M${pa.x} ${pa.y}L${pb.x} ${pb.y}`,
        }),
      );
    }
  svg.append(
    svgElement("path", {
      class: "chart-axis",
      fill: "none",
      d: `M${vertices.map((p) => p.join(" ")).join("L")}Z`,
    }),
  );
  for (let i = 0; i < 3; i++) {
    const label = svgElement("text", {
      class: "chart-text",
      x: vertices[i][0],
      y: vertices[i][1] + (i === 0 ? -20 : 30),
      "text-anchor": "middle",
    });
    label.textContent = `${result[`comp${i + 1}_name`]} · ${comps[i]}`;
    svg.append(label);
  }
  const tieLayer = svgElement("g"),
    dotLayer = svgElement("g");
  svg.append(tieLayer, dotLayer);
  for (const sample of result.samples) {
    const phases =
      result.chart_type === "TERNARY_LLE"
        ? sample.status === "lle"
          ? [sample.x1, sample.x2]
          : []
        : [
            sample.vapor_fraction > 1e-8 ? sample.y : null,
            sample.liquid1_fraction > 1e-8 ? sample.x1 : null,
            sample.liquid2_fraction > 1e-8 ? sample.x2 : null,
          ].filter(Boolean);
    if (phases.length > 1) {
      const coords = phases.map(point);
      tieLayer.append(
        svgElement("path", {
          d: `M${coords.map((p) => `${p.x} ${p.y}`).join("L")}${coords.length === 3 ? "Z" : ""}`,
          fill: coords.length === 3 ? `${colors[2]}33` : "none",
          stroke: colors[0],
          opacity: 0.3,
          "stroke-width": 0.8,
        }),
      );
      for (const composition of phases) {
        const p = point(composition);
        tieLayer.append(
          svgElement("circle", { cx: p.x, cy: p.y, r: 2, fill: colors[0] }),
        );
      }
    }
    const z = point(sample.z),
      failed = sample.status === "failed",
      split = phases.length > 1;
    const dot = svgElement("circle", {
      cx: z.x,
      cy: z.y,
      r: 3,
      fill: failed
        ? failure
        : phases.length === 3
          ? colors[2]
          : split
            ? colors[0]
            : neutral,
      tabindex: 0,
      role: "button",
      "aria-label": `Composition ${comps.map((c) => `${c} ${formatNumber(sample.z[c])}`).join(", ")}; ${sample.status}`,
    });
    const title = svgElement("title");
    title.textContent = `${comps.map((c) => `${c}: ${formatNumber(sample.z[c])}`).join(" · ")}\n${sample.status}`;
    dot.append(title);
    const inspect = () => {
      const detail = element("details", { class: "result-detail" });
      detail.open = true;
      detail.append(
        element("summary", {}, "Selected composition and equilibrated phases"),
        element("pre", {}, JSON.stringify(sample, null, 2)),
      );
      $("phase-point-detail").replaceChildren(detail);
    };
    dot.onclick = inspect;
    dot.onkeydown = (e) => {
      if (e.key === "Enter") inspect();
    };
    dotLayer.append(dot);
  }
  $("chart-frame").replaceChildren(svg);
  $("chart-title").textContent = comps
    .map((c, i) => result[`comp${i + 1}_name`])
    .join(" / ");
  $("chart-legend").replaceChildren();
  for (const [label, color] of [
    ["No split found", neutral],
    ["Two-phase split / tie lines", colors[0]],
    ["Three-phase region", colors[2]],
    ["Unconverged", failure],
  ]) {
    const item = element("span"),
      dot = element("i", { class: "legend-dot" });
    dot.style.background = color;
    item.append(dot, document.createTextNode(label));
    $("chart-legend").append(item);
  }
  $("chart-information").replaceChildren(
    element(
      "p",
      { class: "form-notice" },
      `${result.method} · ${result.temperature_C} °C${result.chart_type === "TERNARY_VLLE" ? ` · ${result.pressure_bar} bar` : ""} · ${result.samples.length} sampled compositions. ${result.chart_type === "TERNARY_LLE" ? "Liquid activity equilibrium; vapor is excluded." : ""} Tie lines and endpoints are sampled, not an interpolated binodal. Click a point for phase compositions and residuals.`,
    ),
  );
  showDiagnostics(result);
  $("export-chart").disabled = false;
  $("export-chart-svg").disabled = false;
}
async function track(identifier) {
  activeJob = identifier;
  writeLocal("pfdsim.chart-job.v1", identifier);
  $("generate-chart").disabled = true;
  $("cancel-chart").hidden = false;
  try {
    const job = await pollJob(identifier, (job) => {
      $("chart-status").textContent =
        job.error || job.progress.at(-1) || `${job.status}…`;
    });
    activeJob = null;
    writeLocal("pfdsim.chart-job.v1", null);
    if (job.status === "completed") {
      data = job.output;
      plot(data);
      writeLocal("pfdsim.chart-result.v1", identifier);
      $("chart-status").textContent = "Diagram complete.";
    } else if (job.status === "failed") throw new Error(job.error);
    else $("chart-status").textContent = "Calculation cancelled.";
  } catch (error) {
    $("chart-status").textContent = error.message;
    if (error.status === 404) {
      activeJob = null;
      writeLocal("pfdsim.chart-job.v1", null);
      $("chart-status").textContent =
        "This calculation is unavailable or expired. You can generate a new diagram.";
      return;
    }
    if (activeJob) {
      const retry = element("button", {}, "Reconnect to calculation");
      retry.onclick = guarded(() => track(identifier));
      $("chart-status").append(document.createElement("br"), retry);
    }
    throw error;
  } finally {
    $("generate-chart").disabled = !!activeJob;
    $("cancel-chart").hidden = !activeJob;
  }
}
function updateForm() {
  const type = $("chart-type").value,
    ternary = type.startsWith("TERNARY");
  $("third-component-field").hidden = !ternary;
  $("third-reference-button").hidden = !ternary;
  $("comp3").required = ternary;
  $("comp3").disabled = !ternary;
  $("pressure-field").hidden = type === "PXY" || type === "TERNARY_LLE";
  $("chart-pressure").disabled = $("pressure-field").hidden;
  $("temperature-field").hidden = ["TXY", "XY", "VLLE"].includes(type);
  $("chart-temperature").disabled = $("temperature-field").hidden;
  $("binodal-minimum-field").hidden = type !== "VLLE";
  $("binodal-minimum-temperature").disabled = type !== "VLLE";
  $("chart-points").min = ternary ? 4 : 10;
  $("chart-points").max = ternary ? 30 : 200;
  $("chart-points").value = ternary ? 12 : 50;
  $("point-count-label").textContent = ternary
    ? "Triangle grid intervals (4–30)"
    : "Curve intervals (10–200)";
}
$("chart-type").onchange = updateForm;
$("chart-form").onsubmit = guarded(async (e) => {
  e.preventDefault();
  if (activeJob) throw new Error("A chart calculation is already active.");
  $("generate-chart").disabled = true;
  $("chart-status").textContent = "Submitting calculation…";
  try {
    // The saved model is the applied definition. A source draft may be invalid
    // or describe different chemistry, and must never be applied implicitly.
    const definition = project
      ? await api("/api/serialize", { pfd: project.pfd })
      : null;
    const response = await api("/api/vle-chart", {
      comp1: $("comp1").value,
      comp2: $("comp2").value,
      comp3: $("comp3").value,
      method: $("chart-method").value,
      chart_type: $("chart-type").value,
      pressure: $("chart-pressure").disabled
        ? 1
        : $("chart-pressure").valueAsNumber,
      temperature: $("chart-temperature").disabled
        ? 25
        : $("chart-temperature").valueAsNumber,
      n_points: $("chart-points").valueAsNumber,
      minimum_temperature: $("binodal-minimum-temperature").disabled
        ? 25 : $("binodal-minimum-temperature").valueAsNumber,
      online_lookup: $("chart-online").checked,
      ...(definition
        ? { text: definition.text, scope: $("chart-scope").value }
        : {}),
      thermo_options: modelOptions.value().thermo_options,
    });
    await track(response.job_id);
  } finally {
    if (!activeJob) $("generate-chart").disabled = false;
  }
});
$("cancel-chart").onclick = guarded(async () => {
  if (activeJob) await api(`/api/jobs/${activeJob}/cancel`, {});
});
$("export-chart").onclick = () =>
  download(
    JSON.stringify(data, null, 2),
    "phase-diagram.json",
    "application/json",
  );
$("export-chart-svg").onclick = () => {
  const svg = $("chart-frame").querySelector("svg").cloneNode(true);
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const [,, width, height] = svg.getAttribute("viewBox").split(" ").map(Number);
  const legend = [...$("chart-legend").children];
  const totalHeight = height + 70 + legend.length * 20;
  svg.setAttribute("viewBox", `0 -60 ${width} ${totalHeight}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", totalHeight);
  const background = getComputedStyle($("chart-frame")).backgroundColor;
  svg.prepend(svgElement("rect", {
    x: 0, y: -60, width, height: totalHeight, fill: background,
  }));
  const style = svgElement("style");
  const textColor = getComputedStyle($("chart-frame").querySelector(".chart-text")).fill;
  const axisColor = getComputedStyle($("chart-frame").querySelector(".chart-axis")).stroke;
  const gridColor = getComputedStyle($("chart-frame").querySelector(".chart-grid")).stroke;
  style.textContent = `.chart-text{fill:${textColor};font:11px monospace}.chart-grid{stroke:${gridColor};stroke-width:.6}.chart-axis{fill:none;stroke:${axisColor};stroke-width:1}.chart-line{fill:none;stroke-width:2}`;
  svg.prepend(style);
  const title = svgElement("text", { x: 20, y: -38, class: "chart-text" });
  title.textContent = $("chart-title").textContent;
  const conditions = svgElement("text", { x: 20, y: -18, class: "chart-text" });
  const atTemperature = ["PXY", "TERNARY_LLE", "TERNARY_VLLE"].includes(data.chart_type);
  const atPressure = data.chart_type !== "PXY" && data.chart_type !== "TERNARY_LLE";
  conditions.textContent = [
    data.chart_type, data.method,
    ...(atTemperature ? [`${data.temperature_C} °C`] : []),
    ...(atPressure ? [`${data.pressure_bar} bar`] : []),
    `Scope: ${data.scope || "global"}`,
  ].join(" · ");
  svg.append(title, conditions);
  legend.forEach((item, index) => {
    const y = height + 15 + index * 20;
    svg.append(svgElement("path", {
      d: `M20 ${y}H38`, stroke: item.querySelector("i").style.backgroundColor,
      "stroke-width": 3,
    }));
    const label = svgElement("text", { x: 46, y: y + 4, class: "chart-text" });
    label.textContent = item.textContent;
    svg.append(label);
  });
  download(
    new XMLSerializer().serializeToString(svg),
    "phase-diagram.svg",
    "image/svg+xml",
  );
};
function scopeModel() {
  const scope = $("chart-scope").value;
  const method =
    scope === "global"
      ? project?.pfd.metadata.thermo_method
      : project?.pfd.thermo_scopes.find((s) => s.name === scope)?.method;
  if (method) $("chart-method").value = method;
  setModelOptions();
}
function setModelOptions() {
  const scoped = project && $("chart-scope").value !== "global";
  const saved =
    project &&
    $("chart-scope").value === "global" &&
    $("chart-method").value === project.pfd.metadata.thermo_method
      ? project.pfd.metadata.thermo_options
      : {};
  modelOptions = structuredEditor(
    { thermo_options: scoped ? {} : saved || {} },
    {
      customEditors: {
        thermo_options: (value, update) =>
          thermoOptionEditor(value, update, $("chart-method").value, scoped),
      },
    },
  );
  $("chart-model-options").replaceChildren(modelOptions.node);
}
function selectSource(value) {
  if (!project) independentOnline = $("chart-online").checked;
  const library = readLocal("pfdsim.laboratories.v1", {});
  project =
    value === "independent"
      ? null
      : value === "imported"
        ? importedProject
        : library[value];
  $("scope-field").hidden = !project;
  $("chart-online").disabled = !!project;
  $("chart-online").checked = project
    ? project.pfd.metadata.online_lookup
    : independentOnline;
  $("chart-scope").replaceChildren(
    element("option", { value: "global" }, "Global"),
  );
  if (project) {
    for (const scope of project.pfd.thermo_scopes)
      $("chart-scope").append(
        element("option", { value: scope.name }, scope.name),
      );
    $("definition-description").textContent =
      `Using applied .pfd component definitions, property correlations, interaction overrides, and lookup policy.${project.pending ? " Unapplied source edits are excluded." : ""}`;
    scopeModel();
  } else {
    $("definition-description").textContent =
      "Enter any chemical identifier supported by .pfd. Reference suggestions are optional.";
    setModelOptions();
  }
  if (referencesLoaded) renderReferences();
}
function renderReferences() {
  const query = $("reference-search").value.trim().toLowerCase();
  const local = (project?.pfd.components || []).map((c) => ({
    identifier: c.symbol,
    name: c.name,
    formula: c.formula || "",
    CAS: c.CAS,
    source: "This laboratory",
    aliases: [c.symbol],
  }));
  const matches = [...local, ...chemicalOptions].filter((c) =>
    [c.name, c.formula, c.CAS, ...(c.aliases || [])]
      .join(" ")
      .toLowerCase()
      .includes(query),
  );
  $("reference-chemical").replaceChildren();
  for (const c of matches.slice(0, 200))
    $("reference-chemical").append(
      element(
        "option",
        { value: c.identifier },
        `${c.name}${c.CAS ? " · " + c.CAS : ""} (${c.source})`,
      ),
    );
  $("reference-status").textContent =
    `${matches.length} matching references${matches.length > 200 ? " · showing the first 200; refine the search" : ""}. You can also type an unlisted identifier.`;
  document.querySelectorAll("[data-reference-target]").forEach((b) => {
    b.disabled = !matches.length;
  });
}
$("reference-search").oninput = renderReferences;
$("reference-browser").ontoggle = guarded(async () => {
  if (!$("reference-browser").open || referencesLoaded) return;
  $("reference-status").textContent = "Loading local reference names…";
  const response = await api("/api/vle-chart/components");
  chemicalOptions = response.components;
  referencesLoaded = true;
  renderReferences();
});
document.querySelectorAll("[data-reference-target]").forEach((button) => {
  button.onclick = () => {
    const input = $(button.dataset.referenceTarget);
    input.value = $("reference-chemical").value;
    $("reference-browser").open = false;
    input.focus();
  };
});
$("chart-source").onchange = () => selectSource($("chart-source").value);
$("chart-scope").onchange = scopeModel;
$("chart-method").onchange = setModelOptions;
$("atlas-open-pfd").onclick = () => $("atlas-pfd-file").click();
$("atlas-pfd-file").onchange = guarded(async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  importedProject = await api("/api/upload", form);
  if (!$("chart-source").querySelector('[value="imported"]'))
    $("chart-source").append(
      element("option", { value: "imported" }, "Imported .pfd"),
    );
  $("chart-source").value = "imported";
  selectSource("imported");
});
async function init() {
  await initializeLibrary();
  config = await api("/api/config");
  for (const method of config.chart_methods)
    $("chart-method").append(element(
      "option", { value: method, ...(method === "STEAM" ? { disabled: "" } : {}) },
      method === "STEAM" ? "STEAM · water only; choose a mixture model" : method,
    ));
  $("chart-method").value = "UNIFAC";
  for (const lab of Object.values(readLocal("pfdsim.laboratories.v1", {})))
    $("chart-source").append(
      element(
        "option",
        { value: lab.id },
        lab.pfd.metadata.process_name || "Untitled laboratory",
      ),
    );
  selectSource("independent");
  updateForm();
  const saved = readLocal("pfdsim.chart-job.v1", null);
  if (saved) await track(saved);
  else {
    const last = readLocal("pfdsim.chart-result.v1", null);
    if (last) {
      try {
        const { job } = await api(`/api/jobs/${last}`);
        if (job.status === "completed") {
          data = job.output;
          plot(data);
          $("chart-status").textContent =
            "Restored your last completed diagram.";
        }
      } catch (error) {
        if (error.status !== 404) throw error;
      }
    }
  }
}
init().catch((error) => toast(error.message, true));
