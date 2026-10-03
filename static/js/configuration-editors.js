import { element, clone } from "./common.js";
import { structuredEditor, labelFor } from "./fields.js";
import { parameterControls } from "./unit-controls.js";

export const componentLabels = {
  symbol: "Flowsheet symbol",
  name: "Chemical identifier (name, formula, CAS or SMILES)",
  molecular_weight: "Molar mass (g/mol)",
  CAS: "CAS registry number",
  smiles: "SMILES structure",
  Tc: "Critical temperature (K)",
  Pc: "Critical pressure (bar)",
  Vc: "Critical molar volume (cm³/mol)",
  Zc: "Critical compressibility factor",
  omega: "Acentric factor",
  Tb: "Normal boiling temperature (K)",
  Tm: "Melting temperature (K)",
  Tt: "Triple-point temperature (K)",
  Pt: "Triple-point pressure (bar)",
  Hf: "Gas formation enthalpy (kJ/mol)",
  Gf: "Gas formation Gibbs energy (kJ/mol)",
  S: "Gas standard entropy (J/mol·K)",
  Hf_liquid: "Liquid formation enthalpy (kJ/mol)",
  Gf_liquid: "Liquid formation Gibbs energy (kJ/mol)",
  S_liquid: "Liquid standard entropy (J/mol·K)",
  Hf_solid: "Solid formation enthalpy (kJ/mol)",
  Gf_solid: "Solid formation Gibbs energy (kJ/mol)",
  S_solid: "Solid standard entropy (J/mol·K)",
  Hvap: "Heat of vaporization (kJ/mol)",
  Hfus: "Heat of fusion (kJ/mol)",
  Hcomb: "Net heat of combustion (kJ/mol)",
  Hcomb_gross: "Gross heat of combustion (kJ/mol)",
  Cp_liquid: "Liquid heat capacity (J/mol·K)",
  Cp_solid: "Solid heat capacity (J/mol·K)",
  Cp_coeffs: "Ideal-gas heat-capacity coefficients",
  rho: "Reference liquid density (kg/m³)",
  rho_T: "Density reference temperature (K)",
  rho_solid: "Solid density (kg/m³)",
  Vm_solid: "Solid molar volume (m³/kmol)",
  particle_diameter: "Representative particle diameter (m)",
  particle_sphericity: "Particle sphericity",
  particle_size_distribution: "Particle-size distribution",
  uniquac_r: "UNIQUAC relative molecular volume (r)",
  uniquac_q: "UNIQUAC relative surface area (q)",
  unifac_groups: "UNIFAC group counts",
  dipole_moment: "Dipole moment (Debye)",
  radius_of_gyration: "Radius of gyration (Å)",
  modified_radius_of_gyration: "Modified radius of gyration (Å)",
  henry_Hcp: "Henry solubility constant at 298.15 K (mol/m³·Pa)",
  henry_B: "Henry temperature coefficient (K)",
  henry_Tmin: "Henry lower temperature bound (K)",
  henry_Tmax: "Henry upper temperature bound (K)",
  henry_Vinf: "Aqueous infinite-dilution molar volume (cm³/mol)",
  henry_Vinf_uncertainty: "Molar-volume uncertainty (cm³/mol)",
  antoine_A: "Antoine coefficient A",
  antoine_B: "Antoine coefficient B",
  antoine_C: "Antoine coefficient C",
  antoine_Tmin: "Antoine lower temperature bound (K)",
  antoine_Tmax: "Antoine upper temperature bound (K)",
  property_correlations: "Temperature-dependent property correlations",
  phase_behavior: "Participation in phase equilibrium",
  solid_material_form: "Solid material form",
  solid_polymorph: "Solid polymorph",
  vapor_dimerization: "Vapor association / dimerization",
  critical_properties_unavailable: "Critical properties are unavailable",
  mc_c1: "Mathias–Copeman EOS alpha coefficient c1",
  mc_c2: "Mathias–Copeman EOS alpha coefficient c2",
  mc_c3: "Mathias–Copeman EOS alpha coefficient c3",
  kappa1: "PRSV alpha coefficient κ1",
  kappa2: "PRSV2 alpha coefficient κ2",
  kappa3: "PRSV2 alpha coefficient κ3",
  twu_l: "Twu EOS alpha coefficient L",
  twu_m: "Twu EOS alpha coefficient M",
  twu_n: "Twu EOS alpha coefficient N",
  twu_c: "Twu EOS volume-translation coefficient",
  hoc_eta: "Hayden–O’Connell self-association parameter η",
  phase_at_STP: "Reference phase at standard conditions",
};
function textField(
  parent,
  label,
  value,
  change,
  { choices = null, help = "", type = "text" } = {},
) {
  const row = element("label", { class: "field" });
  row.append(element("span", { class: "field-label" }, label));
  const input = choices
    ? element("select")
    : element("input", { type, autocomplete: "off" });
  if (type === "number") input.setAttribute("step", "any");
  if (choices) {
    const options = [...choices];
    if (value != null && !options.includes(String(value)))
      options.unshift(String(value));
    for (const option of options)
      input.append(
        element("option", { value: option }, option || "None / global"),
      );
  }
  input.value = value ?? "";
  input.oninput = () => change(input.value);
  input.onchange = input.oninput;
  row.append(input);
  if (help) row.append(element("span", { class: "field-help" }, help));
  parent.append(row);
  return input;
}
function dictionaryControls(value, definitions, update) {
  const schema = definitions.map((d) => ({
    name: d.name,
    canonical: d.name.toLowerCase(),
    section: "operation",
    label: d.label || labelFor(d.name),
    type: d.type || "text",
    default: d.default ?? null,
    values: d.choices,
    description: d.help,
    unit: null,
  }));
  return parameterControls(
    Object.entries(value || {}).map(([name, v]) => ({
      name,
      value: v,
      unit: null,
    })),
    schema,
    "operation",
    (params) =>
      update(
        Object.fromEntries(
          params.map((p) => {
            const definition = definitions.find((d) => d.name === p.name);
            let v = p.value;
            if (
              definition?.type === "number" &&
              v !== "" &&
              Number.isFinite(Number(v))
            )
              v = Number(v);
            return [p.name, v];
          }),
        ),
      ),
  );
}
export function modelOptionEditor(
  value,
  update,
  definitions,
  description = "",
) {
  const box = element("div");
  if (description)
    box.append(element("p", { class: "field-help" }, description));
  box.append(dictionaryControls(value, definitions, update));
  return box;
}
export function thermoOptionEditor(value, update, method, scoped = false) {
  if (scoped || !method?.endsWith("-BV"))
    return element(
      "p",
      { class: "field-help" },
      scoped
        ? "Named scopes use model defaults for vapor correlations. Global model options do not apply here."
        : "This thermodynamic model has no additional options.",
    );
  return modelOptionEditor(
    value,
    update,
    [
      {
        name: "correlation",
        label: "Second-virial vapor correlation",
        choices: ["TSONOPOULOS", "HOC"],
      },
    ],
    "Unchecked controls use model defaults.",
  );
}
export function compositionEditor(
  value,
  update,
  components,
  basis,
  updateBasis,
) {
  const draft = clone(value || {}),
    box = element("div");
  textField(box, "Composition basis", basis || "mole", updateBasis, {
    choices: ["mole", "mass"],
  });
  box.append(
    element(
      "p",
      { class: "field-help" },
      "Enter fractions for the declared components. The simulator normalizes them; an unspecified composition stays unset.",
    ),
  );
  if (!components.length)
    box.append(
      element(
        "p",
        { class: "form-notice" },
        "Declare chemicals in Process settings → Components first.",
      ),
    );
  for (const component of components) {
    textField(
      box,
      `${component.symbol} · ${component.name}`,
      draft[component.symbol] ?? "",
      (v) => {
        if (v === "") delete draft[component.symbol];
        else draft[component.symbol] = Number(v);
        update(Object.keys(draft).length ? clone(draft) : null);
      },
      { type: "number" },
    );
  }
  return box;
}
export function streamSelectionEditor(value, update, streams) {
  const selected = new Set(value || []),
    box = element("div");
  box.append(
    element(
      "p",
      { class: "field-help" },
      "Leave unselected for automatic tear selection. Select internal streams to guide recycle initialization.",
    ),
  );
  for (const stream of streams.filter(
    (s) => !s.source.is_feed && !s.destination.is_product,
  )) {
    const label = element("label", { class: "check-field" }),
      input = element("input", { type: "checkbox" });
    input.checked = selected.has(stream.id);
    input.onchange = () => {
      if (input.checked) selected.add(stream.id);
      else selected.delete(stream.id);
      update([...selected]);
    };
    label.append(
      input,
      element(
        "span",
        {},
        `${stream.id} · ${stream.source.unit_id} → ${stream.destination.unit_id}`,
      ),
    );
    box.append(label);
  }
  return box;
}

export function reactionEditor(
  record,
  update,
  config,
  { components = [], definitions = [], unitType = "", named = false } = {},
) {
  const draft = clone(record),
    box = element("div"),
    parameters = element("div");
  const publish = () => update(clone(draft));
  if (named)
    textField(
      box,
      "Reaction name",
      draft.name,
      (value) => {
        draft.name = value;
        publish();
      },
      { help: "Use this name to reuse the reaction in unit operations." },
    );
  if (!named && definitions.length)
    textField(
      box,
      "Named reaction",
      draft.reference,
      (value) => {
        draft.reference = value || null;
        if (value) {
          draft.equation = "";
          draft.parameters = {};
          const inherited =
            definitions.find((r) => r.name === value)?.parameters || {};
          mode =
            "conversion" in inherited
              ? "conversion"
              : inherited.kinetics_type ||
                inherited.type ||
                (["PFR", "CSTR", "BatchReactor", "PackedBedReactor"].includes(
                  unitType,
                )
                  ? "power_law"
                  : unitType === "Reactor"
                    ? "conversion"
                    : "thermodynamic");
          equationInput.value = "";
        }
        equationInput.disabled = !!draft.reference;
        modeInput.disabled = !!draft.reference;
        modeInput.value = mode;
        renderParameters();
        publish();
      },
      {
        choices: ["", ...definitions.map((r) => r.name)],
        help: "Select a reusable definition, or leave blank to enter an equation below.",
      },
    );
  const equationInput = textField(
    box,
    "Reaction equation",
    draft.equation,
    (value) => {
      draft.equation = value;
      publish();
    },
    {
      help: "Use flowsheet component symbols and stoichiometric coefficients. Use -> for irreversible or <=> for reversible reactions.",
    },
  );
  let mode = "thermodynamic";
  const p = {
    ...(definitions.find((r) => r.name === draft.reference)?.parameters || {}),
    ...(draft.parameters || {}),
  };
  if ("conversion" in p || unitType === "Reactor") mode = "conversion";
  else if (
    p.kinetics_type ||
    p.type ||
    ["PFR", "CSTR", "BatchReactor", "PackedBedReactor"].includes(unitType)
  )
    mode = p.kinetics_type || p.type || "power_law";
  mode =
    {
      "power-law": "power_law",
      powerlaw: "power_law",
      "custom-net": "custom_net",
    }[mode] || mode;
  const modeInput = textField(
    box,
    "Reaction specification",
    mode,
    (value) => {
      mode = value;
      draft.parameters =
        value === "conversion"
          ? { conversion: "" }
          : value === "thermodynamic"
            ? {}
            : {
                kinetics_type: value,
                A: "",
                Ea: 0,
                Ea_unit: "J/mol",
                rate_basis: "concentration",
                concentration_unit: "kmol/m3",
                pressure_unit: "bar",
                rate_unit:
                  unitType === "PackedBedReactor"
                    ? "kmol/kg_cat/h"
                    : "kmol/m3/h",
                ...(value !== "power_law" ? { expression: "" } : {}),
              };
      renderParameters();
      publish();
    },
    {
      choices: ["thermodynamic", "conversion", ...config.kinetics.models],
      help: "Thermodynamic reactions use equilibrium properties. Conversion fixes the reacted fraction. Kinetic models require explicit rate units.",
    },
  );
  box.append(parameters);
  function renderParameters() {
    parameters.replaceChildren();
    if (mode === "thermodynamic") {
      parameters.append(
        element(
          "p",
          { class: "field-help" },
          "No rate parameters are required for a thermodynamic equilibrium reaction. Named definitions can be reused by equilibrium reactors.",
        ),
      );
      if (Object.keys(draft.parameters || {}).length)
        parameters.append(
          dictionaryControls(draft.parameters, [], (value) => {
            draft.parameters = value;
            publish();
          }),
        );
      return;
    }
    const fields =
      mode === "conversion"
        ? [
            {
              name: "conversion",
              label: "Inlet-basis conversion fraction (0–1)",
              type: "number",
              help: "Fraction of the limiting reactant converted by this reaction.",
            },
          ]
        : [
            {
              name: "A",
              label: "Pre-exponential factor A",
              type: "number",
              help: "A uses the declared rate basis and units, including reaction orders.",
            },
            { name: "Ea", label: "Activation energy", type: "number" },
            {
              name: "Ea_unit",
              label: "Activation-energy units",
              choices: config.kinetics.Ea_unit,
            },
            {
              name: "rate_basis",
              label: "Rate basis",
              choices: config.kinetics.basis,
            },
            {
              name: "concentration_unit",
              label: "Concentration units",
              choices: config.kinetics.concentration_unit,
            },
            {
              name: "pressure_unit",
              label: "Pressure / fugacity units",
              choices: config.kinetics.pressure_unit,
            },
            {
              name: "rate_unit",
              label: "Reaction-rate units",
              choices: config.kinetics.rate_unit,
            },
            ...(mode === "power_law"
              ? components.map((c) => ({
                  name: "order_" + c.symbol,
                  label: "Reaction order · " + c.symbol,
                  type: "number",
                  help: "Leave unspecified to use reactant stoichiometric orders.",
                }))
              : [
                  {
                    name: "expression",
                    label: "Rate expression",
                    help: 'Use T, R and component maps C, p, f, a; for example C["A"] uses the declared concentration units.',
                  },
                ]),
          ];
    const typeKey =
      "type" in (draft.parameters || {}) ? "type" : "kinetics_type";
    const inherited =
      definitions.find((r) => r.name === draft.reference)?.parameters || {};
    fields.forEach((field) => {
      if (field.name in inherited) {
        field.default = inherited[field.name];
        field.help = "Unchecked values inherit the named reaction definition.";
      }
    });
    parameters.append(
      dictionaryControls(
        Object.fromEntries(
          Object.entries(draft.parameters || {}).filter(
            ([key]) => !["type", "kinetics_type"].includes(key),
          ),
        ),
        fields,
        (value) => {
          draft.parameters =
            mode === "conversion" ? value : { [typeKey]: mode, ...value };
          publish();
        },
      ),
    );
  }
  modeInput.value = mode;
  modeInput.disabled = !!draft.reference;
  equationInput.disabled = !!draft.reference;
  renderParameters();
  return box;
}
export function interactionEditor(
  record,
  update,
  config,
  { components = [], scopes = [], estimation = false } = {},
) {
  const draft = clone(record),
    box = element("div"),
    parameters = element("div"),
    publish = () => update(clone(draft));
  const choices = ["", ...components.map((c) => c.symbol)];
  textField(
    box,
    "Component 1",
    draft.component1,
    (v) => {
      draft.component1 = v || null;
      publish();
    },
    {
      choices,
      help: estimation
        ? "Leave both components blank for a global estimation policy."
        : "Choose a component declared in this laboratory.",
    },
  );
  textField(
    box,
    "Component 2",
    draft.component2,
    (v) => {
      draft.component2 = v || null;
      publish();
    },
    { choices },
  );
  textField(
    box,
    "Interaction model",
    draft.model,
    (v) => {
      draft.model = v;
      renderParameters();
      publish();
    },
    {
      choices: estimation
        ? ["NRTL", "UNIQUAC"]
        : Object.keys(config.configuration_fields.interaction_models),
    },
  );
  textField(
    box,
    "Thermodynamic scope",
    draft.scope,
    (v) => {
      draft.scope = v || null;
      publish();
    },
    { choices: ["", "global", ...scopes.map((s) => s.name)] },
  );
  box.append(parameters);
  function renderParameters() {
    parameters.replaceChildren();
    const names = estimation
      ? config.configuration_fields.interaction_estimation_fields
      : config.configuration_fields.interaction_models[draft.model] || [];
    const labels = {
      alpha: "NRTL nonrandomness α",
      alpha12: "NRTL nonrandomness α",
      a12: "Interaction 1 → 2 (cal/mol)",
      a21: "Interaction 2 → 1 (cal/mol)",
      kij: "Binary interaction coefficient kᵢⱼ",
      comment: "Source / comments",
      Tmin_K: "Lower temperature bound (K)",
      Tmax_K: "Upper temperature bound (K)",
      tmin_k: "Lower temperature bound (K)",
      tmax_k: "Upper temperature bound (K)",
    };
    const fields = names.map((name) => ({
      name,
      label: labels[name] || labelFor(name),
      type: [
        "comment",
        "source",
        "extrapolation",
        "policy",
        "parameter_order",
      ].includes(name)
        ? "text"
        : "number",
      choices:
        name === "extrapolation"
          ? [
              "unrestricted",
              "clamp",
              "constant_inverse",
              "inverse_linear_quadratic",
              "inverse_square_cubic",
            ]
          : null,
    }));
    parameters.append(
      element(
        "p",
        { class: "field-help" },
        "Leave a coefficient unchecked to retain model/dataset behavior. Scalar NRTL/UNIQUAC energy parameters use cal/mol; direct tau parameters are available below.",
      ),
      dictionaryControls(draft.parameters || {}, fields, (value) => {
        draft.parameters = value;
        publish();
      }),
    );
  }
  renderParameters();
  return box;
}
export function particleEditor(value, update) {
  let draft = clone(value || {}),
    model =
      draft.distribution ||
      draft.model ||
      (draft.diameters_m ? "discrete" : "");
  const box = element("div"),
    content = element("div"),
    publish = () => update(clone(draft));
  textField(
    box,
    "Particle distribution model",
    model,
    (v) => {
      model = v;
      draft =
        v === "discrete"
          ? { diameters_m: [], fractions: [], basis: "mole" }
          : v === "lognormal"
            ? {
                distribution: v,
                d50_m: "",
                geometric_standard_deviation: "",
                basis: "mole",
                classes: 20,
              }
            : v === "weibull"
              ? {
                  distribution: v,
                  scale_diameter_m: "",
                  shape: "",
                  basis: "mole",
                  classes: 20,
                }
              : {};
      render();
      publish();
    },
    { choices: ["", "discrete", "lognormal", "weibull"] },
  );
  box.append(content);
  function render() {
    content.replaceChildren();
    if (!model) return;
    textField(
      content,
      "Fraction basis",
      draft.basis,
      (v) => {
        draft.basis = v;
        publish();
      },
      { choices: ["mole", "mass", "volume", "number"] },
    );
    if (model === "discrete") {
      const table = element("table", { class: "data-table" }),
        head = element("tr");
      ["Diameter (m)", "Fraction", ""].forEach((t) =>
        head.append(element("th", {}, t)),
      );
      table.append(head);
      (draft.diameters_m || []).forEach((diameter, index) => {
        const row = element("tr");
        for (const [name, values] of [
          ["diameters_m", draft.diameters_m],
          ["fractions", draft.fractions],
        ]) {
          const cell = element("td"),
            input = element("input", {
              type: "number",
              step: "any",
              "aria-label": `${name === "diameters_m" ? "Diameter" : "Fraction"} ${index + 1}`,
            });
          input.value = values[index];
          input.oninput = () => {
            values[index] = input.valueAsNumber;
            publish();
          };
          cell.append(input);
          row.append(cell);
        }
        const cell = element("td"),
          remove = element("button", { type: "button" }, "Remove");
        remove.onclick = () => {
          draft.diameters_m.splice(index, 1);
          draft.fractions.splice(index, 1);
          render();
          publish();
        };
        cell.append(remove);
        row.append(cell);
        table.append(row);
      });
      const scroll = element("div", { class: "table-scroll" });
      scroll.append(table);
      content.append(scroll);
      const add = element("button", { type: "button" }, "Add size class");
      add.onclick = () => {
        draft.diameters_m.push("");
        draft.fractions.push("");
        render();
        publish();
      };
      content.append(
        add,
        element(
          "p",
          { class: "field-help" },
          "Use increasing positive diameters in meters. Fractions are normalized by the simulator.",
        ),
      );
    } else {
      const fields =
        model === "lognormal"
          ? [
              ["d50_m", "Median diameter d50 (m)"],
              [
                "geometric_standard_deviation",
                "Geometric standard deviation (≥ 1)",
              ],
            ]
          : [
              ["scale_diameter_m", "Characteristic diameter d63.2 (m)"],
              ["shape", "Weibull / Rosin–Rammler shape"],
            ];
      fields.push(["classes", "Number of size classes"]);
      for (const [key, label] of fields)
        textField(
          content,
          label,
          draft[key],
          (v) => {
            draft[key] = v === "" ? "" : Number(v);
            publish();
          },
          { type: "number" },
        );
    }
  }
  render();
  return box;
}
export function correlationEditor(value, update, config) {
  const draft = clone(value || {}),
    box = element("div"),
    list = element("div"),
    publish = () => update(clone(draft));
  const names = {
    Psat: "Vapor pressure",
    Hvap: "Heat of vaporization",
    Cpl: "Liquid heat capacity",
    Cpg: "Ideal-gas heat capacity",
    Cps: "Solid heat capacity",
    rhol: "Liquid density",
    rhos: "Solid density",
    mul: "Liquid viscosity",
    mug: "Gas viscosity",
    kl: "Liquid thermal conductivity",
    kg: "Gas thermal conductivity",
    sigma: "Surface tension",
    surface_tension: "Surface tension",
  };
  function render() {
    list.replaceChildren();
    for (const [key, correlation] of Object.entries(draft)) {
      const group = element("details", { class: "structured-group" });
      group.open = true;
      group.append(element("summary", {}, names[key] || key));
      textField(
        group,
        "Correlation equation",
        correlation.equation,
        (v) => {
          correlation.equation = v;
          fields();
          publish();
        },
        { choices: config.configuration_fields.property_equations[key] || [] },
      );
      const controls = element("div");
      group.append(controls);
      function fields() {
        controls.replaceChildren();
        const definitions = [
          {
            name: "Tmin_K",
            label: "Lower temperature bound (K)",
            type: "number",
          },
          {
            name: "Tmax_K",
            label: "Upper temperature bound (K)",
            type: "number",
          },
          ...(
            config.configuration_fields.property_coefficients[
              correlation.equation
            ] || []
          ).map((name) => ({
            name,
            label: "Coefficient " + name,
            type: "number",
          })),
          { name: "source", label: "Source / citation" },
        ];
        controls.append(
          dictionaryControls(
            Object.fromEntries(
              Object.entries(correlation).filter(
                ([name]) => name !== "equation",
              ),
            ),
            definitions,
            (v) => {
              const next = { equation: correlation.equation, ...v };
              Object.keys(correlation).forEach((k) => delete correlation[k]);
              Object.assign(correlation, next);
              publish();
            },
          ),
        );
      }
      fields();
      const remove = element(
        "button",
        { type: "button", class: "danger" },
        "Remove correlation",
      );
      remove.onclick = () => {
        delete draft[key];
        render();
        publish();
      };
      group.append(remove);
      list.append(group);
    }
  }
  box.append(
    element(
      "p",
      { class: "field-help" },
      "Explicit overrides replace property lookup. Coefficient units and equations follow the .pfd property reference.",
    ),
    list,
  );
  const addRow = element("div", { class: "add-field" }),
    select = element("select", { "aria-label": "Property to correlate" });
  for (const key of Object.keys(config.configuration_fields.property_equations))
    select.append(element("option", { value: key }, names[key] || key));
  const add = element("button", { type: "button" }, "Add correlation");
  add.onclick = () => {
    if (select.value in draft) return;
    draft[select.value] = {
      equation: config.configuration_fields.property_equations[select.value][0],
    };
    render();
    publish();
  };
  addRow.append(select, add);
  box.append(addRow);
  render();
  return box;
}
