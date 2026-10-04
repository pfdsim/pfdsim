// Observation editing uses one append implementation for pasted and manual data.
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function nextId(rows) {
  return String(Math.max(0, ...rows.map(row => /^\d+$/.test(String(row.id)) ? Number(row.id) : 0)) + 1);
}
export function mergeSigma(current, overrides) {
  if (!Object.keys(overrides).length) return structuredClone(current ?? {});
  const base = typeof current === "number" ? Object.fromEntries(["log_fugacity", "log_gamma", "HE_J_mol", "curvature", "third_derivative"].map(key => [key, current])) : current || {};
  return { ...base, ...overrides };
}
export function repairObservationIds(existing) {
  const rows = structuredClone(existing), renames = {};
  const used = new Set();
  for (const row of rows) {
    const old = String(row.id ?? "");
    if (!old || uuid.test(old) || used.has(old)) {
      row.id = nextId(rows); renames[old] = row.id;
      if (!row.source && row.group === old) row.group = row.id;
    } else row.id = old;
    used.add(row.id);
  }
  return { rows, renames };
}
export function appendObservations(existing, incoming, sigma = {}) {
  const rows = structuredClone(existing), added = [], sourceIds = [];
  for (const original of incoming) {
    const row = structuredClone(original), old = String(row.id ?? "");
    // Meaningful named source IDs can be retained; generated numeric/UUID IDs
    // become short, unique numbers across successive imports.
    row.id = old && !uuid.test(old) && !/^\d+$/.test(old) && !rows.some(item => item.id === old) ? old : nextId(rows);
    if (!row.source && row.group === old) row.group = row.id;
    row.sigma = mergeSigma(row.sigma, sigma);
    rows.push(row); added.push(row.id); sourceIds.push({ source_id: old, observation_id: row.id });
  }
  return { rows, added, sourceIds };
}
