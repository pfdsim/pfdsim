export const $ = (id) => document.getElementById(id);
export const clone = (value) => structuredClone(value);
export function element(tag, attributes = {}, text = null) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  if (text !== null) node.textContent = text;
  return node;
}
export function svgElement(tag, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, value);
  return node;
}
export function toast(message, error = false) {
  const node = element(
    "div",
    { class: `toast${error ? " error" : ""}` },
    message,
  );
  $("notifications").append(node);
  setTimeout(() => node.remove(), error ? 10000 : 5000);
}
let sessionState = null,
  sessionRequest = null;
export async function getSession(refresh = false) {
  if (!refresh && sessionState) return sessionState;
  if (!sessionRequest)
    sessionRequest = api("/api/session").finally(() => {
      sessionRequest = null;
    });
  return sessionRequest;
}
export async function api(path, data, options = {}) {
  const headers =
    data instanceof FormData ? {} : { "Content-Type": "application/json" };
  if (data !== undefined)
    headers["X-CSRF-Token"] = (await getSession()).csrf_token;
  const response = await fetch(path, {
    ...options,
    method: data === undefined ? "GET" : "POST",
    headers,
    body:
      data === undefined
        ? undefined
        : data instanceof FormData
          ? data
          : JSON.stringify(data),
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(
      `The server returned an unreadable response (${response.status}).`,
    );
  }
  if (!response.ok) {
    const error = new Error(
      result.error || `Request failed (${response.status}).`,
    );
    error.status = response.status;
    throw error;
  }
  if (result.csrf_token) {
    sessionState = result;
    renderAccountStatus();
  }
  return result;
}
export function modal(title, content) {
  $("modal-title").textContent = title;
  $("modal-content").replaceChildren(content);
  if (!$("modal").open) $("modal").showModal();
}
$("modal-close").onclick = () => $("modal").close();
export function confirmAction(title, message, action, label = "Continue") {
  const box = element("div");
  box.append(element("p", {}, message));
  const buttons = element("div", { class: "form-actions" });
  const cancel = element("button", {}, "Cancel");
  cancel.onclick = () => $("modal").close();
  const confirm = element("button", { class: "primary" }, label);
  confirm.onclick = () => {
    $("modal").close();
    action();
  };
  buttons.append(cancel, confirm);
  box.append(buttons);
  modal(title, box);
}
export function download(text, filename, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = element("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function readLocal(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
export function writeLocal(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export const settingsDefaults = {
  theme: "dark",
  grid: true,
  snap: true,
  motion: !matchMedia("(prefers-reduced-motion: reduce)").matches,
  max_iterations: 100,
  tolerance: 0.0001,
  thermo_method: "IDEAL",
  fluid_phase_model: "VLE",
  online_lookup: true,
  allow_computation: true,
};
export const settings = {
  ...settingsDefaults,
  ...readLocal("pfdsim.settings.v1", {}),
};
export function applySettings() {
  document.body.dataset.theme = settings.theme;
  document.body.dataset.grid = settings.grid ? "on" : "off";
  document.body.dataset.motion = settings.motion ? "on" : "off";
}
applySettings();
export function tabs(container, names, active, onSelect) {
  container.replaceChildren();
  for (const [key, label] of Object.entries(names)) {
    const button = element(
      "button",
      {
        class: key === active ? "active" : "",
        "aria-pressed": String(key === active),
      },
      label,
    );
    button.onclick = () => onSelect(key);
    container.append(button);
  }
}
export async function pollJob(identifier, onUpdate, { signal } = {}) {
  while (!signal?.aborted) {
    const { job } = await api(
      `/api/jobs/${encodeURIComponent(identifier)}`,
      undefined,
      { signal },
    );
    onUpdate(job);
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      getSession(true).catch(() => {});
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}
export function guarded(action) {
  return async (...args) => {
    try {
      await action(...args);
    } catch (error) {
      toast(error.message, true);
    }
  };
}
export function formatNumber(value, digits = 5) {
  if (value === null || value === undefined || !Number.isFinite(Number(value)))
    return "—";
  return Number(value).toLocaleString(undefined, {
    maximumSignificantDigits: digits,
    notation:
      Math.abs(Number(value)) !== 0 &&
      (Math.abs(Number(value)) < 0.0001 || Math.abs(Number(value)) >= 10000000)
        ? "scientific"
        : "standard",
  });
}

function renderAccountStatus() {
  if (!sessionState || !$("account-button")) return;
  $("account-button").textContent =
    sessionState.user?.username || "Guest · sign in";
  const remaining = sessionState.quota.remaining_seconds;
  $("compute-budget").textContent =
    `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s CPU left`;
  $("compute-budget").title =
    `Backend compute only. ${sessionState.quota.limit_seconds / 60} minutes per day. Resets at 00:00 UTC.`;
}
function accountDialog(active = "login") {
  const box = element("div");
  const description = element(
    "p",
    {},
    "Guest: 5 CPU minutes per day. Account: 15 CPU minutes per day, with laboratories autosaved to your account. Budgets reset at 00:00 UTC. Waiting and web requests do not count.",
  );
  box.append(description);
  if (sessionState?.user) {
    box.append(element("h3", {}, `Signed in as ${sessionState.user.username}`));
    const logout = element("button", {}, "Sign out");
    logout.onclick = guarded(async () => {
      await api("/api/account/logout", {});
      location.reload();
    });
    box.append(logout);
  } else {
    const nav = element("nav", { class: "tabs" }),
      content = element("div");
    box.append(nav, content);
    const show = (mode) => {
      tabs(nav, { login: "Sign in", register: "Create account" }, mode, show);
      content.replaceChildren();
      const form = element("form");
      const username = element("input", {
          required: "",
          minlength: "3",
          maxlength: "32",
          autocomplete: "username",
          name: "username",
        }),
        password = element("input", {
          required: "",
          type: "password",
          minlength: mode === "register" ? "10" : "1",
          maxlength: "256",
          autocomplete:
            mode === "register" ? "new-password" : "current-password",
          name: "password",
        });
      for (const [title, input] of [
        ["Username", username],
        ["Password", password],
      ]) {
        const label = element("label", { class: "field" });
        label.append(element("span", { class: "field-label" }, title), input);
        form.append(label);
      }
      if (mode === "register")
        form.append(
          element(
            "p",
            { class: "field-help" },
            "Use 3–32 letters, numbers, dots, underscores, or hyphens for your username, and at least 10 characters for your password. No email is needed yet.",
          ),
        );
      const error = element("p", {
          class: "form-notice error-notice",
          hidden: "",
        }),
        submit = element(
          "button",
          { class: "primary", type: "submit" },
          mode === "login" ? "Sign in" : "Create account",
        );
      form.append(error, submit);
      form.onsubmit = async (e) => {
        e.preventDefault();
        submit.disabled = true;
        error.hidden = true;
        try {
          await api(`/api/account/${mode}`, {
            username: username.value,
            password: password.value,
          });
          location.reload();
        } catch (failure) {
          error.textContent = failure.message;
          error.hidden = false;
          submit.disabled = false;
        }
      };
      content.append(form);
    };
    show(active);
  }
  modal("Laboratory account", box);
}
if ($("account-button"))
  $("account-button").onclick = guarded(async () => {
    await getSession(true);
    accountDialog();
  });
getSession().catch((error) => toast(error.message, true));
