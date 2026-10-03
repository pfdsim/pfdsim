import {
  $,
  api,
  getSession,
  readLocal,
  writeLocal,
  clone,
  toast,
} from "./common.js";

const pending = new Map();
// Each page edits the revision it opened. localStorage is shared by tabs and
// may contain a newer tab's save metadata while this page still has old input.
const baselines = new Map();
let timer,
  running = false;
function signature(value) {
  function sorted(item) {
    if (Array.isArray(item)) return item.map(sorted);
    if (item && typeof item === "object")
      return Object.fromEntries(
        Object.keys(item)
          .sort()
          .map((key) => [key, sorted(item[key])]),
      );
    return item;
  }
  return JSON.stringify(sorted(documentFor(value)));
}
function documentFor(saved) {
  return Object.fromEntries(
    ["pfd", "text", "pending", "filename", "job", "lastJob", "layout"].map(
      (key) => [key, saved?.[key] ?? null],
    ),
  );
}
export async function initializeLibrary() {
  try {
    await mergeLibrary();
  } catch (error) {
    status("Account sync unavailable · browser drafts available");
    toast(
      `Could not load account laboratories. Your browser drafts remain available. ${error.message}`,
      true,
    );
  } finally {
    for (const saved of Object.values(readLocal("pfdsim.laboratories.v1", {})))
      rememberBaseline(saved);
  }
}
function rememberBaseline(saved) {
  if (
    !baselines.has(saved.id) ||
    (saved.cloudSignature && signature(saved) === saved.cloudSignature)
  )
    baselines.set(saved.id, {
      owner: saved.cloudOwner,
      version: saved.cloudVersion || 0,
    });
}
async function mergeLibrary() {
  const session = await getSession();
  if (!session.user) return;
  const { flowsheets } = await api("/api/flowsheets"),
    library = readLocal("pfdsim.laboratories.v1", {});
  for (const remote of flowsheets) {
    const local = library[remote.id],
      cloud = {
        ...remote.document,
        id: remote.id,
        updated: remote.updated,
        cloudOwner: session.user.id,
        cloudVersion: remote.version,
      };
    if (
      !local ||
      signature(local) === signature(cloud) ||
      signature(local) === local.cloudSignature
    ) {
      cloud.cloudSignature = signature(cloud);
      library[remote.id] = cloud;
      baselines.set(remote.id, { owner: session.user.id, version: remote.version });
    } else if (
      local.cloudOwner === session.user.id &&
      (baselines.get(remote.id)?.version ?? local.cloudVersion) !== remote.version
    ) {
      local.cloudConflict = cloud;
    }
  }
  writeLocal("pfdsim.laboratories.v1", library);
}
export function queueCloud(saved) {
  rememberBaseline(saved);
  pending.set(saved.id, clone(saved));
  status("Account sync pending…");
  clearTimeout(timer);
  timer = setTimeout(flushCloud, 900);
}
async function flushCloud() {
  if (running) return;
  running = true;
  try {
    const session = await getSession();
    if (!session.user) {
      pending.clear();
      status("Browser storage only · sign in for account saves");
      return;
    }
    for (const [id, saved] of pending) {
      const library = readLocal("pfdsim.laboratories.v1", {}),
        local = library[id] || saved,
        baseline = baselines.get(id);
      if (local.cloudConflict) {
        pending.delete(id);
        status("Cloud conflict · local draft preserved", id);
        continue;
      }
      const sameOwner = local.cloudOwner === session.user.id;
      if (local.cloudOwner && !sameOwner) {
        pending.delete(id);
        status("Saved locally · use a new copy for this account", id);
        continue;
      }
      if (sameOwner && local.cloudSignature === signature(saved)) {
        pending.delete(id);
        status("Saved to your account", id);
        continue;
      }
      status("Saving to your account…", id);
      try {
        const result = await api(`/api/flowsheets/${encodeURIComponent(id)}`, {
          version: sameOwner ? baseline.version : 0,
          document: documentFor(saved),
        });
        const latest = readLocal("pfdsim.laboratories.v1", {});
        baselines.set(id, { owner: session.user.id, version: result.version });
        if (latest[id])
          Object.assign(latest[id], {
            cloudOwner: session.user.id,
            cloudVersion: result.version,
            cloudSignature: signature(saved),
          });
        writeLocal("pfdsim.laboratories.v1", latest);
        if (signature(pending.get(id)) === signature(saved)) pending.delete(id);
        status(
          pending.has(id) ? "Account sync pending…" : "Saved to your account",
          id,
        );
      } catch (error) {
        if (error.status === 409) {
          await mergeLibrary();
          pending.delete(id);
          status("Cloud conflict · local draft preserved", id);
          toast(error.message, true);
        } else {
          status("Saved locally · account sync offline", id);
          break;
        }
      }
    }
  } catch (error) {
    status("Saved locally · account sync offline");
  } finally {
    running = false;
    if (pending.size && navigator.onLine) timer = setTimeout(flushCloud, 10000);
  }
}
function status(message, id = null) {
  if (id && readLocal("pfdsim.last.v1", null) !== id) return;
  if ($("cloud-save-status")) $("cloud-save-status").textContent = message;
}
window.addEventListener("online", flushCloud);
