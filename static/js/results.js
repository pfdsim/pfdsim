import { element, svgElement, tabs, formatNumber, download } from "./common.js";
import { labelFor } from "./fields.js";

function metricLabel(key) {
  const names = {
    T_in_C: "Inlet temperature (°C)",
    T_out_C: "Outlet temperature (°C)",
    P_in_bar: "Inlet pressure (bar)",
    P_out_bar: "Outlet pressure (bar)",
    P_drop_bar: "Pressure drop (bar)",
    duty_kW: "Heat duty (kW)",
    stage_T: "Stage temperature (K)",
    stage_P: "Stage pressure (bar)",
    N_stages: "Number of stages",
    reflux_ratio: "Reflux ratio (L/D)",
    D_to_F: "Distillate / feed ratio",
  };
  return names[key] || labelFor(key);
}

const valueText = (value) =>
  value == null
    ? "—"
    : typeof value === "number"
      ? formatNumber(value, 7)
      : typeof value === "boolean"
        ? value
          ? "Yes"
          : "No"
        : String(value);
export const percentage = (value) =>
  typeof value === "number" ? `${formatNumber(value * 100, 5)}%` : "—";

function table(headers, rows) {
  const scroll = element("div", { class: "table-scroll" }),
    node = element("table", { class: "data-table" });
  const head = element("thead"),
    heading = element("tr");
  headers.forEach((label) => heading.append(element("th", {}, label)));
  head.append(heading);
  node.append(head);
  const body = element("tbody");
  node.append(body);
  scroll.append(node);
  const controls = element("div", { class: "result-pagination" });
  let page = 0;
  const redraw = () => {
    body.replaceChildren();
    for (const cells of rows.slice(page * 100, (page + 1) * 100)) {
      const row = element("tr");
      cells.forEach((cell) => {
        const td = element("td");
        td.append(
          cell instanceof Node
            ? cell
            : document.createTextNode(valueText(cell)),
        );
        row.append(td);
      });
      body.append(row);
    }
    controls.replaceChildren();
    if (rows.length > 100) {
      const previous = element("button", {}, "Previous"),
        next = element("button", {}, "Next");
      previous.disabled = !page;
      next.disabled = (page + 1) * 100 >= rows.length;
      previous.onclick = () => {
        page--;
        redraw();
      };
      next.onclick = () => {
        page++;
        redraw();
      };
      controls.append(
        previous,
        element(
          "span",
          {},
          `${page * 100 + 1}–${Math.min((page + 1) * 100, rows.length)} of ${rows.length}`,
        ),
        next,
      );
    }
  };
  redraw();
  scroll.append(controls);
  return scroll;
}
function detail(title, build, open = false) {
  const node = element("details", { class: "result-detail" });
  node.append(element("summary", {}, title));
  let ready = false;
  node.ontoggle = () => {
    if (node.open && !ready) {
      ready = true;
      node.append(build());
    }
  };
  node.open = open;
  if (open) {
    ready = true;
    node.append(build());
  }
  return node;
}
function profile(values, title) {
  const box = element("div"),
    finite = values.filter((v) => typeof v === "number" && Number.isFinite(v));
  if (finite.length > 1) {
    const lo = Math.min(...finite),
      hi = Math.max(...finite),
      span = hi - lo || 1;
    const x = (i) => 55 + (i * 610) / Math.max(1, values.length - 1),
      y = (v) => 210 - ((v - lo) * 165) / span;
    const svg = svgElement("svg", {
      viewBox: "0 0 700 255",
      class: "result-profile",
      role: "img",
      "aria-label": title,
    });
    svg.append(
      svgElement("path", {
        d: "M55 35V210H665",
        class: "chart-axis",
        fill: "none",
      }),
    );
    let d = "",
      connected = false;
    values.forEach((v, i) => {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        connected = false;
        return;
      }
      d += `${connected ? "L" : "M"}${x(i)} ${y(v)} `;
      connected = true;
    });
    svg.append(
      svgElement("path", { d, class: "chart-line", stroke: "#78e6cb" }),
    );
    for (const [label, px, py, anchor] of [
      [valueText(lo), 45, 215, "end"],
      [valueText(hi), 45, 45, "end"],
      ["1", 55, 235, "middle"],
      [String(values.length), 665, 235, "middle"],
    ]) {
      const text = svgElement("text", {
        x: px,
        y: py,
        class: "chart-text",
        "text-anchor": anchor,
      });
      text.textContent = label;
      svg.append(text);
    }
    box.append(svg);
  }
  box.append(
    table(
      ["Stage / point", title],
      values.map((value, i) => [i + 1, value]),
    ),
  );
  return box;
}
export function readableData(data, name = "Details") {
  const box = element("div");
  if (data == null || typeof data !== "object") {
    box.append(element("p", {}, valueText(data)));
    return box;
  }
  if (Array.isArray(data)) {
    if (!data.length) {
      box.append(element("p", { class: "field-help" }, "No entries."));
      return box;
    }
    if (data.every((v) => v == null || typeof v === "number"))
      return profile(data, name);
    if (
      data.every(
        (v) =>
          v &&
          typeof v === "object" &&
          !Array.isArray(v) &&
          Object.values(v).every(
            (item) => item == null || typeof item !== "object",
          ),
      )
    ) {
      const keys = [...new Set(data.flatMap(Object.keys))];
      box.append(
        table(
          ["Entry", ...keys.map(metricLabel)],
          data.map((v, i) => [i + 1, ...keys.map((key) => v[key])]),
        ),
      );
      return box;
    }
    if (data.every((v) => v == null || typeof v !== "object")) {
      box.append(
        table(
          ["Entry", name],
          data.map((v, i) => [i + 1, v]),
        ),
      );
      return box;
    }
    data.forEach((v, i) =>
      box.append(detail(`${name} · ${i + 1}`, () => readableData(v, name))),
    );
    return box;
  }
  const scalars = Object.entries(data).filter(
    ([, v]) => v == null || typeof v !== "object",
  );
  if (scalars.length)
    box.append(
      table(
        ["Property", "Value"],
        scalars.map(([key, v]) => [metricLabel(key), v]),
      ),
    );
  for (const [key, value] of Object.entries(data))
    if (value && typeof value === "object")
      box.append(
        detail(metricLabel(key), () => readableData(value, metricLabel(key))),
      );
  return box;
}
function composition(stream) {
  const components = [
    ...new Set(
      [
        stream.composition,
        stream.x1 || stream.x,
        stream.x2,
        stream.y,
        stream.solid_composition,
      ]
        .filter(Boolean)
        .flatMap(Object.keys),
    ),
  ];
  const phases = [
    ["Overall", stream.composition],
    ["Liquid 1", stream.x1 || stream.x],
    ["Liquid 2", stream.x2],
    ["Vapor", stream.y],
    ["Solid", stream.solid_composition],
  ].filter(([, values]) => values && Object.keys(values).length);
  return table(
    ["Component", ...phases.map(([name]) => `${name} (mol %)`)],
    components.map((c) => [c, ...phases.map(([, v]) => percentage(v[c]))]),
  );
}
function streamDetail(stream) {
  const box = element("div");
  box.append(
    element("h3", {}, "Phase distribution"),
    table(
      ["Phase", "Fraction"],
      Object.entries(stream.phase_fractions || {}).map(([key, v]) => [
        labelFor(key),
        percentage(v),
      ]),
    ),
    element("h3", {}, "Component compositions"),
    composition(stream),
  );
  const properties = {};
  for (const [key, label] of [
    ["T_C", "Temperature (°C)"],
    ["P", "Pressure (bar)"],
    ["F", "Molar flow (kmol/h)"],
    ["mass_flow", "Mass flow (kg/h)"],
    ["H", "Molar enthalpy (kJ/kmol)"],
    ["S", "Molar entropy (kJ/kmol·K)"],
    ["rho", "Molar density (kmol/m³)"],
    ["Cp", "Heat capacity (kJ/kmol·K)"],
    ["mu", "Dynamic viscosity (Pa·s)"],
    ["MW", "Molar mass (kg/kmol)"],
  ])
    if (key in stream) properties[label] = stream[key];
  box.append(
    element("h3", {}, "Physical properties"),
    table(["Property", "Value"], Object.entries(properties)),
  );
  const shown = new Set([
    "T",
    "T_C",
    "P",
    "F",
    "mass_flow",
    "H",
    "S",
    "rho",
    "Cp",
    "mu",
    "MW",
    "composition",
    "x",
    "x1",
    "x2",
    "y",
    "solid_composition",
    "phase_fractions",
  ]);
  box.append(
    detail("Additional stream details", () =>
      readableData(
        Object.fromEntries(
          Object.entries(stream).filter(([key]) => !shown.has(key)),
        ),
      ),
    ),
  );
  return box;
}
export function renderSimulationData(container, data) {
  const results = data.results,
    streams = results.streams || {},
    units = results.units || {};
  const nav = element("nav", {
      class: "tabs",
      "aria-label": "Simulation result sections",
    }),
    content = element("div");
  container.append(nav, content);
  function show(tab) {
    tabs(
      nav,
      {
        streams: `Streams · ${Object.keys(streams).length}`,
        equipment: `Equipment · ${Object.keys(units).length}`,
        diagnostics: "Solver diagnostics",
      },
      tab,
      show,
    );
    content.replaceChildren();
    if (tab === "streams") {
      content.append(
        element(
          "p",
          { class: "field-help" },
          "Click a stream to inspect its compositions, phase distribution, and physical properties.",
        ),
      );
      const details = element("div");
      const rows = Object.entries(streams).map(([id, s]) => {
        const button = element("button", { class: "result-link" }, id);
        button.onclick = () => {
          details.replaceChildren(element("h3", {}, id), streamDetail(s));
          details.scrollIntoView({ block: "nearest", behavior: "auto" });
        };
        return [
          button,
          s.T_C ?? (typeof s.T === "number" ? s.T - 273.15 : null),
          s.P,
          s.F,
          s.mass_flow,
          percentage(s.vapor_fraction),
          labelFor(s.phase_status || "Calculated"),
        ];
      });
      content.append(
        table(
          [
            "Stream",
            "Temperature (°C)",
            "Pressure (bar)",
            "Flow (kmol/h)",
            "Mass (kg/h)",
            "Vapor",
            "Phase",
          ],
          rows,
        ),
        details,
      );
    } else if (tab === "equipment") {
      content.append(
        element(
          "p",
          { class: "field-help" },
          "Positive heat adds energy; positive shaft work acts on the process. Select equipment for its performance and profiles.",
        ),
      );
      const details = element("div");
      const rows = Object.entries(units).map(([id, unit]) => {
        const button = element("button", { class: "result-link" }, id);
        button.onclick = () => {
          details.replaceChildren(
            element("h3", {}, id),
            readableData(unit.performance || {}, "Performance"),
          );
          if (unit.warnings?.length)
            details.append(readableData(unit.warnings, "Warnings"));
          details.append(
            detail("Calculated outlet streams", () => {
              const box = element("div");
              Object.entries(unit.outlets || {}).forEach(([port, stream]) =>
                box.append(detail(labelFor(port), () => streamDetail(stream))),
              );
              return box;
            }),
          );
          if (unit.unrepresented_enthalpy_change)
            details.append(
              table(
                ["Inventory enthalpy change (kJ/h)"],
                [[unit.unrepresented_enthalpy_change]],
              ),
            );
        };
        return [
          button,
          unit.heat_duty_kW ?? unit.heat_duty / 3600,
          unit.work_kW ?? unit.work / 3600,
          Object.keys(unit.outlets || {}).length,
          unit.warnings?.length || 0,
        ];
      });
      content.append(
        table(
          [
            "Equipment",
            "Heat duty (kW)",
            "Shaft work (kW)",
            "Outlets",
            "Warnings",
          ],
          rows,
        ),
        details,
      );
    } else {
      const recycle = results.recycle_info || {};
      content.append(
        readableData(
          {
            method: recycle.method,
            tear_streams: recycle.tear_streams || [],
            failed_evaluations: recycle.failed_evaluations || 0,
            worst_error: recycle.worst_error,
          },
          "Recycle",
        ),
      );
      content.append(
        detail("Detailed recycle diagnostics", () => readableData(recycle)),
        detail("Thermodynamic scope corrections", () =>
          readableData(results.thermo_scope_corrections || []),
        ),
      );
      const exportButton = element("button", {}, "Export full results as JSON");
      exportButton.onclick = () =>
        download(
          JSON.stringify(results, null, 2),
          "simulation-results.json",
          "application/json",
        );
      content.append(exportButton);
    }
  }
  show("streams");
}
