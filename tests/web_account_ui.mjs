// Exercise the actual shared API and account UI without a browser dependency.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { SourceTextModule, createContext } from "node:vm";

const source = await readFile(new URL("../static/js/common.js", import.meta.url), "utf8");
const storage = new Map();
let now = 100_000;
let storageAvailable = true;
let taskStatus = 202;
let taskQuota;
let currentSession;
let usageStatus = 200;
let recentJobs = [];
class Clock extends Date {
  static now() { return now; }
}
function session(user, remaining, limit, grace = 0) {
  return {
    csrf_token: "test-csrf", user,
    quota: {
      remaining_seconds: remaining, limit_seconds: limit, grace_seconds: grace,
      used_seconds: limit === null ? 1200 : limit - remaining,
    },
  };
}
async function loadUI() {
  const nodes = new Map();
  function element(tag = "div") {
    return {
      tag, textContent: "", title: "", dataset: {}, children: [], attributes: {}, listeners: {},
      setAttribute(key, value) { this.attributes[key] = value; },
      addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); },
      dispatch(type, event = {}) { return Promise.all((this.listeners[type] || []).map(fn => fn(event))); },
      contains(target) { return this === target || this.children.some(child => child.contains(target)); },
      focus() { this.focused = true; },
      remove() {},
      append(...children) {
        this.children.push(...children);
        for (const child of children) child.parentElement = this;
      },
      replaceChildren(...children) { this.children = []; this.append(...children); },
      showModal() { this.open = true; },
      close() { this.open = false; },
    };
  }
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  node("compute-budget-menu").append(node("compute-budget-toggle"), node("compute-budget-panel"));
  node("compute-budget-toggle").append(node("compute-budget"));
  node("compute-budget-panel").append(node("compute-usage-summary"), node("compute-usage-list"), node("compute-more-info"));
  const documentEvents = element();
  const context = createContext({
    document: {
      getElementById: node, createElement: element, body: element(),
      addEventListener: documentEvents.addEventListener.bind(documentEvents),
    },
    localStorage: {
      getItem(key) {
        if (!storageAvailable) throw new Error("Storage unavailable");
        return storage.get(key) ?? null;
      },
      setItem(key, value) {
        if (!storageAvailable) throw new Error("Storage unavailable");
        storage.set(key, value);
      },
    },
    fetch: async path => ({
      ok: path === "/api/session" || (path === "/api/usage" ? usageStatus : taskStatus) < 400,
      status: path === "/api/session" ? 200 : path === "/api/usage" ? usageStatus : taskStatus,
      json: async () => path === "/api/session" ? currentSession : path === "/api/usage"
        ? { quota: currentSession.quota, recent_jobs: recentJobs, error: "Usage unavailable" }
        : { job_id: "test-job", status: "queued", quota: taskQuota, error: "No allowance" },
    }),
    matchMedia: () => ({ matches: false }),
    setTimeout: () => 0,
    Date: Clock, FormData,
  });
  const module = new SourceTextModule(source, { context });
  await module.link(() => { throw new Error("Unexpected import"); });
  await module.evaluate();
  await module.namespace.getSession();
  return {
    api: module.namespace.api, getSession: module.namespace.getSession, node,
    dispatchDocument: documentEvents.dispatch.bind(documentEvents),
  };
}
function text(node) { return [node.textContent, ...node.children.map(text)].join(" "); }
function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
const regular = { id: "regular", username: "PFDSim" };
currentSession = session(regular, 900, 900, 15);
let ui = await loadUI();
const warningCount = () => ui.node("notifications").children.length;
assert.equal(ui.node("compute-budget").textContent, "15m 0s CPU left");
assert.match(ui.node("compute-budget").title, /15 additional CPU seconds/);

// The quota disclosure shows private task usage without opening a modal.
recentJobs = [
  { kind: "simulation", status: "completed", created: now / 1000, cpu_seconds: 12.5 },
  { kind: "fit", status: "failed", created: now / 1000 - 10, cpu_seconds: 4 },
];
ui.node("compute-budget-menu").open = true;
await ui.node("compute-budget-menu").dispatch("toggle");
assert.equal(ui.node("modal").open, undefined);
assert.equal(ui.node("compute-usage-list").children.length, 2);
assert.match(text(ui.node("compute-usage-list")), /Simulation.*12.5 CPU s.*completed/s);
assert.match(text(ui.node("compute-usage-list")), /Parameter fitting.*4 CPU s.*failed/s);
assert.match(ui.node("compute-usage-summary").textContent, /Used today: 0 CPU seconds/);
await ui.dispatchDocument("keydown", { key: "Escape" });
assert.equal(ui.node("compute-budget-menu").open, false);
assert.equal(ui.node("compute-budget-toggle").focused, true);
ui.node("compute-budget-menu").open = true;
await ui.dispatchDocument("click", { target: ui.node("compute-more-info") });
assert.equal(ui.node("compute-budget-menu").open, true);
await ui.dispatchDocument("click", { target: ui.node("account-button") });
assert.equal(ui.node("compute-budget-menu").open, false);

// Loading failures and an empty history leave the help button available.
ui.node("compute-budget-menu").open = true;
usageStatus = 503;
await ui.node("compute-budget-menu").dispatch("toggle");
assert.match(ui.node("compute-usage-summary").textContent, /Recent usage is unavailable/);
usageStatus = 200;
recentJobs = [];
await ui.node("compute-budget-menu").dispatch("toggle");
assert.match(text(ui.node("compute-usage-list")), /No recent tasks/);
await ui.node("compute-more-info").onclick();
assert.equal(ui.node("compute-budget-menu").open, false);
assert.equal(ui.node("modal").open, true);
assert.equal(ui.node("modal-title").textContent, "Usage and rate limits");
const info = text(ui.node("modal-content"));
assert.match(info, /PFDSim is free software/);
assert.match(info, /won't charge for subscriptions/);
assert.match(info, /fair usage for everyone/);
assert.match(info, /300 seconds/);
assert.match(info, /900 seconds/);
assert.match(info, /15 additional CPU seconds/);
assert.match(info, /00:00 UTC/);
assert.match(info, /pfdsim@chemicalprocess.org/);
const emailLink = descendants(ui.node("modal-content")).find(node => node.tag === "a");
assert.equal(emailLink.attributes.href, "mailto:pfdsim@chemicalprocess.org");
ui.node("modal").close();

// Fresh submission quota controls the warning, rather than cached session data.
taskQuota = session(regular, 60, 900, 15).quota;
await ui.api("/api/simulate", {});
assert.equal(warningCount(), 0);
taskQuota.remaining_seconds = 59.5;
await ui.api("/api/simulate", {});
assert.equal(warningCount(), 1);
assert.match(ui.node("notifications").children[0].textContent, /Less than 1 CPU minute/);
assert.equal(ui.node("compute-budget").textContent, "0m 59s CPU left");
for (const path of ["/api/vle-chart", "/api/unifac-groups", "/api/fitting",
  "/api/fitting/prefill", "/api/fitting/admin/publish", "/api/fitting/admin/withdraw"]) {
  await ui.api(path, {});
}
assert.equal(warningCount(), 1);
now += 59_999;
await ui.api("/api/fitting", {});
assert.equal(warningCount(), 1);
now += 1;
await ui.api("/api/fitting", {});
assert.equal(warningCount(), 2);

// Cooldown survives navigation/reload; failed starts and session refreshes do not warn.
currentSession = session(regular, 59.5, 900, 15);
ui = await loadUI();
await ui.api("/api/simulate", {});
assert.equal(warningCount(), 0);
now += 60_000;
taskStatus = 429;
await assert.rejects(ui.api("/api/simulate", {}), /No allowance/);
await ui.getSession(true);
assert.equal(warningCount(), 0);
taskStatus = 202;
taskQuota.remaining_seconds = 0;
await ui.api("/api/simulate", {});
assert.equal(warningCount(), 1);

// Each identity has its own cooldown. Guests warn, and unlimited root never does.
currentSession = session({ id: "other", username: "Other" }, 30, 900, 15);
await ui.getSession(true);
taskQuota = currentSession.quota;
await ui.api("/api/fitting", {});
assert.equal(warningCount(), 2);
currentSession = session(null, 30, 300);
await ui.getSession(true);
taskQuota = currentSession.quota;
await ui.api("/api/unifac-groups", {});
assert.equal(warningCount(), 3);
currentSession = session({ id: "root-id", username: "root" }, null, null);
await ui.getSession(true);
taskQuota = currentSession.quota;
await ui.api("/api/fitting/admin/publish", {});
assert.equal(warningCount(), 3);
assert.equal(ui.node("compute-budget").textContent, "Unlimited CPU usage");
assert.match(ui.node("compute-budget").title, /No daily CPU limit/);
await ui.node("account-button").onclick();
assert.match(text(ui.node("modal-content")), /Signed in as root/);
assert.doesNotMatch(text(ui.node("modal-content")), /daily CPU|subscriptions|fair usage/);
ui.node("modal").close();
await ui.node("compute-more-info").onclick();
assert.match(text(ui.node("modal-content")), /unlimited daily CPU usage/);
ui.node("modal").close();

// Sign-up keeps its credential instructions without the rate-limit explanation.
currentSession = session(null, 300, 300);
await ui.node("account-button").onclick();
const createAccount = descendants(ui.node("modal-content")).find(node => node.textContent === "Create account");
createAccount.onclick();
assert.match(text(ui.node("modal-content")), /at least 10 characters/);
assert.doesNotMatch(text(ui.node("modal-content")), /CPU minutes|subscriptions|fair usage/);
ui.node("modal").close();

// A denied browser-storage permission still suppresses repeat warnings in this page.
storageAvailable = false;
currentSession = session(regular, 30, 900, 15);
ui = await loadUI();
taskQuota = currentSession.quota;
await ui.api("/api/simulate", {});
await ui.api("/api/vle-chart", {});
assert.equal(warningCount(), 1);
console.log("Quota dropdown, recent usage, fair-usage info, contact link, account UI, and warning cooldown passed.");
