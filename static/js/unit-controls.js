import { element, clone } from "./common.js";
import { structuredEditor, labelFor } from "./fields.js";

export function parameterControls(parameters, schema, section, onUpdate) {
  const box = element("div"),
    draft = clone(parameters),
    fields = element("div");
  box.append(fields);
  const byName = new Map(schema.map((s) => [s.name.toLowerCase(), s]));
  const canonical = new Map(
    schema
      .filter((s) => s.canonical === s.name.toLowerCase())
      .map((s) => [s.canonical, s]),
  );
  function render() {
    fields.replaceChildren();
    const present = new Set(
      draft.map(
        (p) =>
          byName.get(p.name.toLowerCase())?.canonical || p.name.toLowerCase(),
      ),
    );
    const controls =
      section === "operation"
        ? [...canonical.values()].filter((s) => s.section === section)
        : [];
    const existing = draft.map((p) => ({
      parameter: p,
      info: byName.get(p.name.toLowerCase()) || {
        name: p.name,
        label: labelFor(p.name),
        type: typeof p.value,
        section,
      },
    }));
    const rows = [
      ...existing,
      ...controls
        .filter((s) => !present.has(s.canonical))
        .map((info) => ({ parameter: null, info })),
    ];
    for (const { parameter, info } of rows) {
      const field = element("div", {
        class: "guided-setting",
        "data-setting": info.canonical || info.name,
      });
      const label = element("span", { class: "field-label" }, info.label);
      const inputId = "setting-" + crypto.randomUUID();
      const header = element("div", { class: "guided-setting-header" }),
        enabled = element("input", {
          type: "checkbox",
          "aria-label": `Specify ${info.label}`,
        });
      enabled.checked = !!parameter;
      header.append(label, enabled);
      field.append(header);
      let current = parameter;
      let input, nested;
      const initial = parameter?.value ?? info.default ?? "";
      if (
        (initial && typeof initial === "object") ||
        info.type === "object" ||
        info.type === "array"
      ) {
        const value =
          initial && typeof initial === "object"
            ? initial
            : info.type === "array"
              ? []
              : {};
        nested = structuredEditor({ [info.label]: value });
        input = nested.node;
        const update = () => {
          if (current) {
            current.value = nested.value()[info.label];
            onUpdate(clone(draft));
          }
        };
        input.addEventListener("input", update);
        input.addEventListener("click", () => queueMicrotask(update));
      } else {
        input =
          info.values || info.type === "boolean"
            ? element("select", { id: inputId })
            : element("input", {
                id: inputId,
                type: "text",
                "aria-label": info.label,
              });
        if (info.values || info.type === "boolean") {
          const choices = [...(info.values || ["true", "false"])];
          if (initial !== "" && !choices.map(String).includes(String(initial)))
            choices.unshift(String(initial));
          for (const value of choices)
            input.append(element("option", { value }, String(value)));
        }
        input.value = String(initial);
        input.placeholder =
          info.default != null ? `Default: ${info.default}` : "Not specified";
        label.id = inputId + "-label";
        input.setAttribute("aria-labelledby", label.id);
        input.oninput = () => {
          if (current) {
            current.value = input.value;
            onUpdate(clone(draft));
          }
        };
        input.onchange = input.oninput;
      }
      const row = element("div", { class: "field-row" });
      row.append(input);
      const unit = element("select", { "aria-label": `${info.label} units` });
      const suggestedUnit = (info.unit || "").split(" or ")[0];
      const initialUnit = parameter ? parameter.unit || "" : suggestedUnit;
      const choices = new Set(["", initialUnit, suggestedUnit]);
      if (/^(C|K|F)$/.test(suggestedUnit || initialUnit))
        ["C", "K", "F"].forEach((u) => choices.add(u));
      else if ((suggestedUnit || initialUnit) === "bar")
        ["bar", "kPa", "Pa", "atm", "psi"].forEach((u) => choices.add(u));
      for (const value of choices)
        unit.append(
          element("option", { value }, value || "Default model units"),
        );
      unit.value = initialUnit;
      unit.onchange = () => {
        if (current) {
          current.unit = unit.value || null;
          onUpdate(clone(draft));
        }
      };
      if (initialUnit || suggestedUnit) row.append(unit);
      field.append(row);
      field.append(
        element(
          "p",
          { class: "field-help" },
          info.description ||
            `Default: ${info.default == null ? "determined by the model or other specifications" : info.default}.`,
        ),
      );
      const toggle = () => {
        input
          .querySelectorAll?.("input,select,textarea,button")
          .forEach((node) => {
            node.disabled = !enabled.checked;
          });
        if ("disabled" in input) input.disabled = !enabled.checked;
        unit.disabled = !enabled.checked;
      };
      toggle();
      enabled.onchange = () => {
        if (enabled.checked) {
          current = {
            name: info.name,
            value: nested ? nested.value()[info.label] : input.value,
            unit: unit.value || null,
          };
          draft.push(current);
        } else {
          const index = draft.indexOf(current);
          if (index >= 0) draft.splice(index, 1);
          current = null;
        }
        toggle();
        onUpdate(clone(draft));
      };
      fields.append(field);
    }
  }
  render();
  const add = element("div", { class: "add-field" }),
    choice = element("select", { "aria-label": "Add optional setting" });
  const existingNames = new Set(
    draft.map(
      (p) =>
        byName.get(p.name.toLowerCase())?.canonical || p.name.toLowerCase(),
    ),
  );
  for (const info of canonical.values())
    if (info.section === section && !existingNames.has(info.canonical))
      choice.append(element("option", { value: info.name }, info.label));
  if (section !== "operation" && choice.options.length) {
    const button = element("button", { type: "button" }, "Add setting");
    button.onclick = () => {
      const info = byName.get(choice.value.toLowerCase());
      if (
        draft.some(
          (p) =>
            (byName.get(p.name.toLowerCase())?.canonical ||
              p.name.toLowerCase()) === info.canonical,
        )
      )
        return;
      draft.push({
        name: info.name,
        value: clone(info.default ?? ""),
        unit: info.unit || null,
      });
      onUpdate(clone(draft));
      render();
    };
    add.append(choice, button);
    box.append(add);
  }
  const custom = element("details", { class: "structured-group" });
  custom.append(element("summary", {}, "Custom parameter"));
  const customName = element("input", {
      placeholder: "Parameter name",
      "aria-label": "Custom parameter name",
    }),
    button = element("button", { type: "button" }, "Add");
  const customRow = element("div", { class: "add-field" });
  customRow.append(customName, button);
  custom.append(
    element(
      "p",
      { class: "field-help" },
      "For dynamic outlet specifications or a parameter not listed above. Uses the .pfd syntax.",
    ),
    customRow,
  );
  button.onclick = () => {
    const name = customName.value.trim();
    if (!name || draft.some((p) => p.name.toLowerCase() === name.toLowerCase()))
      return;
    draft.push({ name, value: "", unit: null });
    onUpdate(clone(draft));
    render();
    customName.value = "";
  };
  box.append(custom);
  return box;
}
