import { element, clone, tabs, toast } from "./common.js";

export function labelFor(key) {
  return key
    .replaceAll("_", " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^./, (c) => c.toUpperCase());
}
const forbiddenKeys = new Set(["__proto__", "prototype", "constructor"]);
export function typeDefault(type) {
  if (/dict|object/i.test(type)) return {};
  if (/list|array/i.test(type)) return [];
  if (/bool/i.test(type)) return false;
  if (/float|int|number/i.test(type)) return 0;
  return "";
}

// One recursive editor handles scalar fields, typed dictionaries, arrays and
// arbitrarily nested specifications. No raw-JSON-only fallback is required.
export function structuredEditor(
  value,
  {
    defaults = {},
    types = {},
    enums = {},
    templates = {},
    labels = {},
    help = {},
    recordSections = {},
    suggestions = {},
    open = true,
    customEditors = {},
    recordEditors = {},
    enumLabels = {},
    contextLabels = {},
    contextHelp = {},
    arrayTypes = {},
  } = {},
) {
  const holder = { value: clone(value) };
  const baseLabels = labels,
    baseHelp = help;
  const root = element("div");
  function renderValue(parent, owner, key, name, path, options = {}) {
    const current = owner[key];
    const labels = { ...baseLabels, ...contextLabels[path.split(".")[0]] };
    const help = { ...baseHelp, ...contextHelp[path.split(".")[0]] };
    const recordName = path.split(".").at(-2);
    if (/^\d+$/.test(String(key)) && recordEditors[recordName]) {
      parent.append(
        recordEditors[recordName](current, (updated) => {
          owner[key] = updated;
        }),
      );
      return;
    }
    if (customEditors[name]) {
      parent.append(
        customEditors[name](
          current,
          (updated) => {
            owner[key] = updated;
          },
          holder.value,
        ),
      );
      return;
    }
    const arrayName = path.split(".").at(-3);
    const suggestion = suggestions[arrayName]?.[owner.name];
    const choice =
      (name === "value" ? suggestion?.values : null) ||
      enums[path] ||
      enums[name];
    if (current !== null && typeof current === "object") {
      const array = Array.isArray(current);
      const group = element("details", { class: "structured-group" });
      options.dynamic ??= !Array.isArray(current);
      group.open = options.open ?? open;
      group.append(
        element(
          "summary",
          {},
          `${labels[name] || labelFor(name)}${array ? ` · ${current.length}` : ""}`,
        ),
      );
      const fields = element("div");
      const parentName = path.split(".").at(-2);
      const record = /^\d+$/.test(String(key));
      const known =
        options.defaults ||
        defaults[path] ||
        defaults[name] ||
        (record ? defaults[parentName] || templates[parentName] : null) ||
        {};
      const declaredTypes =
        options.types ||
        types[path] ||
        types[name] ||
        (record ? types[parentName] : null) ||
        {};
      const knownTypes = typeof declaredTypes === "object" ? declaredTypes : {};
      const sections = record ? recordSections[parentName] : null;
      let activeSection = sections ? Object.keys(sections)[0] : null;
      const sectionNav = element("nav", {
        class: "tabs",
        "aria-label": `${parentName} options`,
      });
      if (sections) group.append(sectionNav);
      const rebuild = () => {
        fields.replaceChildren();
        if (sections)
          tabs(
            sectionNav,
            Object.fromEntries(
              Object.entries(sections).map(([key, s]) => [key, s.label]),
            ),
            activeSection,
            (key) => {
              activeSection = key;
              rebuild();
            },
          );
        for (const childKey of Object.keys(current)) {
          if (childKey === "_editor_port_key") continue;
          if (sections && !sections[activeSection].keys.includes(childKey))
            continue;
          const row = element("div");
          if (array) {
            const heading = element("div", { class: "array-header" });
            const item = current[childKey];
            heading.append(
              element(
                "span",
                {},
                `${Number(childKey) + 1} · ${item?.id || item?.symbol || item?.name || "Entry"}`,
              ),
            );
            const remove = element(
              "button",
              { type: "button", class: "danger" },
              "Remove",
            );
            remove.onclick = () => {
              current.splice(Number(childKey), 1);
              rebuild();
            };
            heading.append(remove);
            row.append(heading);
          }
          renderValue(
            row,
            current,
            childKey,
            array ? "Entry" : childKey,
            `${path}.${childKey}`,
          );
          if (!array) {
            const remove = element(
              "button",
              { type: "button", class: "danger" },
              `Remove ${labelFor(childKey)}`,
            );
            remove.style.fontSize = "10px";
            remove.style.marginBottom = "12px";
            remove.onclick = () => {
              delete current[childKey];
              rebuild();
            };
            if (options.dynamic) row.append(remove);
          }
          fields.append(row);
        }
        group.querySelector("summary").textContent =
          `${labels[name] || labelFor(name)}${array ? ` · ${current.length}` : ""}`;
      };
      rebuild();
      group.append(fields);
      const add = element("div", { class: "add-field" });
      const knownKeys = Object.keys(known).filter((k) => !(k in current));
      const selector = element("select", {
        "aria-label": `Field to add to ${labelFor(name)}`,
      });
      if (!array && Object.keys(known).length) {
        const picker = element("select", {
          "aria-label": "Add optional property",
        });
        const refreshChoices = () => {
          picker.replaceChildren();
          for (const key of Object.keys(known).filter(
            (key) =>
              !(key in current) &&
              (!sections || sections[activeSection].keys.includes(key)),
          ))
            picker.append(
              element("option", { value: key }, labels[key] || labelFor(key)),
            );
        };
        refreshChoices();
        const button = element("button", { type: "button" }, "Add property");
        button.onclick = () => {
          const key = picker.value;
          if (!key || key in current) return;
          current[key] = clone(
            known[key] ?? typeDefault(knownTypes[key] || "String"),
          );
          rebuild();
          refreshChoices();
        };
        add.append(picker, button);
        if (sections) {
          const originalRebuild = rebuild;
          sectionNav.addEventListener("click", () => {
            originalRebuild();
            refreshChoices();
          });
        }
      } else if (!array) {
        const keyInput = element("input", {
          placeholder: "Field / component / parameter",
          "aria-label": "New field name",
        });
        const available = element("datalist");
        available.id = `keys-${crypto.randomUUID()}`;
        for (const knownKey of knownKeys)
          available.append(element("option", { value: knownKey }));
        keyInput.setAttribute("list", available.id);
        selector.append(
          ...["String", "Number", "Boolean", "Object", "Array"].map((t) =>
            element("option", { value: t }, t),
          ),
        );
        const button = element("button", { type: "button" }, "Add");
        button.onclick = () => {
          const newKey = keyInput.value.trim();
          if (!newKey || newKey in current || forbiddenKeys.has(newKey)) {
            toast("Choose a new, valid field name.", true);
            return;
          }
          if (
            (options.restricted || Object.keys(knownTypes).length) &&
            !(newKey in known)
          ) {
            toast("Choose a supported field from the suggestions.", true);
            return;
          }
          current[newKey] =
            newKey in known
              ? clone(
                  known[newKey] ?? typeDefault(knownTypes[newKey] || "String"),
                )
              : typeDefault(selector.value);
          if (sections) {
            const section = Object.entries(sections).find(([, s]) =>
              s.keys.includes(newKey),
            );
            if (section) activeSection = section[0];
          }
          keyInput.value = "";
          rebuild();
        };
        add.append(keyInput, available, selector, button);
      } else {
        const template = templates[path] || templates[name];
        if (!template && !arrayTypes[name])
          selector.append(
            ...["String", "Number", "Boolean", "Object", "Array"].map((t) =>
              element("option", { value: t }, t),
            ),
          );
        const button = element("button", { type: "button" }, "＋ Add entry");
        button.onclick = () => {
          current.push(
            template
              ? clone(template)
              : typeDefault(arrayTypes[name] || selector.value),
          );
          rebuild();
        };
        if (!template && !arrayTypes[name]) add.append(selector);
        add.append(button);
      }
      if (array || options.dynamic || Object.keys(known).length)
        group.append(add);
      parent.append(group);
      return;
    }
    const label = element("label", {
      class: typeof current === "boolean" ? "check-field" : "field",
    });
    const caption = element(
      "span",
      { class: "field-label" },
      labels[name] || labelFor(name),
    );
    let input;
    if (current === null) {
      input = element(
        "button",
        {
          type: "button",
          "aria-label": `Set optional ${labels[name] || labelFor(name)}`,
        },
        "Set optional value",
      );
      input.onclick = () => {
        const recordType =
          types[path] ||
          types[name] ||
          types[path.split(".")[0]]?.[name] ||
          "String";
        owner[key] = choice ? choice[0] : typeDefault(recordType);
        label.remove();
        renderValue(parent, owner, key, name, path, { optional: true });
      };
    } else if (choice) {
      input = element("select");
      const choices = [...choice];
      if (current != null && !choices.includes(current))
        choices.unshift(current);
      for (const option of choices)
        input.append(
          element(
            "option",
            { value: option },
            enumLabels[name]?.[option] || option,
          ),
        );
      input.value = current ?? choices[0];
      input.onchange = () => {
        owner[key] = input.value;
        if (name === "thermo_method" && "thermo_options" in owner)
          owner.thermo_options = {};
        if (name === "recycle_method" && "recycle_options" in owner)
          owner.recycle_options = {};
        if (["thermo_method", "recycle_method"].includes(name)) rootRender();
      };
    } else if (typeof current === "boolean") {
      input = element("input", { type: "checkbox" });
      input.checked = current;
      input.onchange = () => {
        owner[key] = input.checked;
      };
    } else {
      input = element(name === "description" ? "textarea" : "input", {
        type: typeof current === "number" ? "number" : "text",
      });
      if (typeof current === "number") input.setAttribute("step", "any");
      input.value = current;
      if (name === "name" && suggestions[arrayName]) {
        const list = element("datalist", {
          id: `params-${crypto.randomUUID()}`,
        });
        Object.entries(suggestions[arrayName]).forEach(([key, info]) =>
          list.append(
            element("option", { value: key, label: info.description || key }),
          ),
        );
        input.setAttribute("list", list.id);
        parent.append(list);
      }
      input.oninput = () => {
        owner[key] =
          typeof current === "number" ? input.valueAsNumber : input.value;
      };
    }
    if (typeof current === "boolean") label.append(input, caption);
    else label.append(caption, input);
    if (help[name])
      label.append(element("span", { class: "field-help" }, help[name]));
    if (
      options.optional ||
      String(types[path] || types[name] || "").includes("Optional")
    ) {
      const clear = element(
        "button",
        { type: "button" },
        "Use default / unset",
      );
      clear.onclick = () => {
        owner[key] = null;
        label.remove();
        renderValue(parent, owner, key, name, path, { optional: true });
      };
      if (current !== null) label.append(clear);
    }
    parent.append(label);
  }
  // Pass the schema only to its matching root; nested arbitrary objects can
  // add keys, whereas fixed records retain their known fields.
  function rootRender() {
    root.replaceChildren();
    const data = holder.value;
    for (const key of Object.keys(data)) {
      const dynamic =
        typeof data[key] === "object" && !Array.isArray(data[key]);
      renderValue(root, data, key, key, key, {
        dynamic,
        defaults: defaults[key],
        types: types[key],
        restricted: typeof types[key] === "object",
      });
    }
  }
  rootRender();
  return { node: root, value: () => clone(holder.value) };
}

export function tabbedForm({
  navigation,
  content,
  sections,
  active,
  value,
  options = {},
  onSave,
  buttonLabel = "Apply changes",
}) {
  let draft = clone(value),
    editor;
  function show(key) {
    if (editor) Object.assign(draft, editor.value());
    active = key;
    tabs(
      navigation,
      Object.fromEntries(
        Object.entries(sections).map(([k, v]) => [k, v.label]),
      ),
      active,
      show,
    );
    const keys = sections[key].keys;
    editor = structuredEditor(
      Object.fromEntries(
        keys.filter((k) => k in draft).map((k) => [k, draft[k]]),
      ),
      typeof options === "function" ? options(draft) : options,
    );
    content.replaceChildren();
    if (sections[key].description)
      content.append(
        element("p", { class: "field-help" }, sections[key].description),
      );
    content.append(editor.node);
    const error = element("div", {
      class: "form-notice error-notice",
      hidden: "",
    });
    const actions = element("div", { class: "form-actions" });
    const save = element("button", { class: "primary" }, buttonLabel);
    save.onclick = async () => {
      save.disabled = true;
      error.hidden = true;
      try {
        Object.assign(draft, editor.value());
        function check(value) {
          if (typeof value === "number" && !Number.isFinite(value))
            throw new Error(
              "Fill in every numeric value, or remove its optional field.",
            );
          if (value && typeof value === "object")
            Object.values(value).forEach(check);
        }
        check(draft);
        await onSave(clone(draft));
      } catch (failure) {
        error.textContent = failure.message;
        error.hidden = false;
      } finally {
        save.disabled = false;
      }
    };
    actions.append(save);
    content.append(error, actions);
  }
  show(active || Object.keys(sections)[0]);
  return { value: () => ({ ...draft, ...editor.value() }) };
}
