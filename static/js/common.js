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
  placeNotifications();
  $("notifications").append(node);
  setTimeout(() => node.remove(), error ? 10000 : 5000);
}
function placeNotifications(){
  const host=$("modal").open?$("modal"):document.body;
  if($("notifications").parentElement!==host)host.append($("notifications"));
}
let sessionState = null,
  sessionRequest = null;
const cpuWarningTimes = new Map();
const CPU_WARNING_COOLDOWN_MS = 60_000;
function warnLowComputeBudget(quota) {
  const remaining = quota.remaining_seconds;
  if (remaining === null || remaining >= 60) return;
  const key = `pfdsim.compute-warning.v1.${sessionState.user?.id || "guest"}`;
  const now = Date.now();
  const last = Math.max(readLocal(key, 0), cpuWarningTimes.get(key) || 0);
  if (now - last < CPU_WARNING_COOLDOWN_MS) return;
  cpuWarningTimes.set(key, now);
  writeLocal(key, now);
  toast("Less than 1 CPU minute remains in your daily allowance.");
}
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
  } else if (result.quota && sessionState) {
    sessionState = { ...sessionState, quota: result.quota };
  }
  if (result.quota) {
    renderAccountStatus();
  }
  if (response.status === 202 && result.job_id && result.quota) {
    warnLowComputeBudget(result.quota);
  }
  return result;
}
export function modal(title, content) {
  $("modal-title").textContent = title;
  $("modal-content").replaceChildren(content);
  if (!$("modal").open) $("modal").showModal();
  placeNotifications();
}
$("modal").addEventListener("close",placeNotifications);
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
  if (sessionState.quota.limit_seconds === null) {
    $("compute-budget").textContent = "Unlimited CPU usage";
    $("compute-budget").title = "Backend compute only. No daily CPU limit.";
    return;
  }
  const remaining = sessionState.quota.remaining_seconds;
  $("compute-budget").textContent =
    `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s CPU left`;
  $("compute-budget").title =
    `Backend compute only. ${sessionState.quota.limit_seconds / 60} minutes per day. Resets at 00:00 UTC.` +
    (sessionState.quota.grace_seconds > 0
      ? ` Running work may use up to ${sessionState.quota.grace_seconds} additional CPU seconds; new tasks require remaining allowance.`
      : "");
}
function rateLimitDialog() {
  const box = element("div");
  box.append(element("p", {}, "PFDSim is free software. I won't charge for subscriptions, but I need to ensure fair usage for everyone."));
  const limits = element("ul");
  for (const text of [
    "Guests get 5 CPU minutes (300 seconds) per day.",
    "Signed-in accounts get 15 CPU minutes (900 seconds) per day.",
    "Signed-in accounts may use up to 15 additional CPU seconds for work already running. New tasks cannot start once the daily allowance is exhausted.",
    "Daily allowances reset at 00:00 UTC.",
  ]) limits.append(element("li", {}, text));
  box.append(limits);
  if (sessionState.quota.limit_seconds === null)
    box.append(element("p", {}, "Your account has unlimited daily CPU usage."));
  box.append(element("p", {}, "Only CPU time spent on backend calculations counts. Waiting and web requests do not count. Each task has a 30-minute time limit."));
  const contact = element("p", {}, "If you need more usage, email ");
  contact.append(
    element("a", { href: "mailto:pfdsim@chemicalprocess.org" }, "pfdsim@chemicalprocess.org"),
    element("span", {}, " to request a higher allowance."),
  );
  box.append(contact);
  modal("Usage and rate limits", box);
}

const computeMenu = $("compute-budget-menu");
if (computeMenu) {
  let requestVersion = 0;
  computeMenu.addEventListener("toggle", async () => {
    if (!computeMenu.open) return;
    const version = ++requestVersion;
    const summary = $("compute-usage-summary"), list = $("compute-usage-list");
    summary.textContent = "Loading recent compute…";
    list.replaceChildren();
    try {
      await getSession(true);
      const { quota, recent_jobs } = await api("/api/usage");
      if (!computeMenu.open || version !== requestVersion) return;
      summary.textContent = `Used today: ${formatNumber(quota.used_seconds)} CPU seconds.`;
      if (!recent_jobs.length) {
        list.append(element("li", {}, "No recent tasks for this account or session."));
      }
      const labels = {
        simulation: "Simulation", chart: "Phase chart", groups: "UNIFAC groups",
        fit: "Parameter fitting", fit_prefill: "Fitting prefill", fit_publish: "Parameter publication",
      };
      for (const job of recent_jobs) {
        const row = element("li"), heading = element("div", { class: "compute-usage-row" });
        heading.append(
          element("span", {}, labels[job.kind] || job.kind),
          element("span", {}, `${formatNumber(job.cpu_seconds)} CPU s`),
        );
        const when = new Date(job.created * 1000).toLocaleString(undefined, {
          month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        });
        row.append(heading, element("span", { class: "compute-usage-detail" }, `${job.status} · ${when}`));
        list.append(row);
      }
    } catch {
      if (computeMenu.open && version === requestVersion)
        summary.textContent = "Recent usage is unavailable. You can still view rate-limit information.";
    }
  });
  document.addEventListener("click", (event) => {
    if (!computeMenu.contains(event.target)) computeMenu.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && computeMenu.open) {
      computeMenu.open = false;
      $("compute-budget-toggle").focus();
    }
  });
  $("compute-more-info").onclick = guarded(async () => {
    computeMenu.open = false;
    await getSession(true);
    rateLimitDialog();
  });
}

function accountDialog(active = "login") {
  const box = element("div");
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
      const setupToken = element("input", { type: "password", autocomplete: "off", name: "setup_token" });
      const tokenLabel = element("label", { class: "field" });
      tokenLabel.append(element("span", { class: "field-label" }, "Root setup token · from the server"), setupToken);
      tokenLabel.hidden = true;
      username.oninput = () => { tokenLabel.hidden = mode !== "register" || username.value.toLowerCase() !== "root"; setupToken.required = !tokenLabel.hidden; };
      for (const [title, input] of [
        ["Username", username],
        ["Password", password],
      ]) {
        const label = element("label", { class: "field" });
        label.append(element("span", { class: "field-label" }, title), input);
        form.append(label);
      }
      form.append(tokenLabel);
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
            ...(mode === "register" && username.value.toLowerCase() === "root" ? { setup_token: setupToken.value } : {}),
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
