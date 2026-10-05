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
class Clock extends Date {
  static now() { return now; }
}
function session(user, remaining, limit, grace = 0) {
  return {
    csrf_token: "test-csrf", user,
    quota: { remaining_seconds: remaining, limit_seconds: limit, grace_seconds: grace },
  };
}
async function loadUI() {
  const nodes = new Map();
  function element() {
    return {
      textContent: "", title: "", dataset: {}, children: [],
      setAttribute() {}, addEventListener() {}, remove() {},
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
  const context = createContext({
    document: { getElementById: node, createElement: element, body: element() },
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
      ok: path === "/api/session" || taskStatus < 400,
      status: path === "/api/session" ? 200 : taskStatus,
      json: async () => path === "/api/session" ? currentSession : {
        job_id: "test-job", status: "queued", quota: taskQuota, error: "No allowance",
      },
    }),
    matchMedia: () => ({ matches: false }),
    setTimeout: () => 0,
    Date: Clock, FormData,
  });
  const module = new SourceTextModule(source, { context });
  await module.link(() => { throw new Error("Unexpected import"); });
  await module.evaluate();
  await module.namespace.getSession();
  return { api: module.namespace.api, getSession: module.namespace.getSession, node };
}
const regular = { id: "regular", username: "PFDSim" };
currentSession = session(regular, 900, 900, 15);
let ui = await loadUI();
const warningCount = () => ui.node("notifications").children.length;
assert.equal(ui.node("compute-budget").textContent, "15m 0s CPU left");
assert.match(ui.node("compute-budget").title, /15 additional CPU seconds/);

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
assert.match(ui.node("modal-content").children[0].children[0].textContent, /unlimited daily CPU usage/);

// A denied browser-storage permission still suppresses repeat warnings in this page.
storageAvailable = false;
currentSession = session(regular, 30, 900, 15);
ui = await loadUI();
taskQuota = currentSession.quota;
await ui.api("/api/simulate", {});
await ui.api("/api/vle-chart", {});
assert.equal(warningCount(), 1);
console.log("Account UI, grace copy, low-budget threshold, task coverage, and warning cooldown passed.");
