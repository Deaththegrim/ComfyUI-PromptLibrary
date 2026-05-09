// Pure helpers for the localStorage tier-3 selection persistence used by
// the gallery widget. Extracted from promptLibrary.js so they can be unit
// tested under Node without a DOM. The browser bundle imports these and
// passes them the browser's localStorage; the test harness injects a shim.

export function lsSelKey(nodeId, propsKey) {
  const id = (nodeId === undefined || nodeId === null) ? "anon" : nodeId;
  const key = propsKey || "pl_state";
  return `pl_sel_${id}_${key}`;
}

export function lsReadSel(storage, nodeId, propsKey) {
  try {
    const raw = storage.getItem(lsSelKey(nodeId, propsKey));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(x => typeof x === "string") : [];
  } catch (_e) {
    return [];
  }
}

export function lsWriteSel(storage, nodeId, propsKey, ids) {
  try {
    const list = (ids || []).filter(x => typeof x === "string" && x.length > 0);
    if (list.length === 0) {
      storage.removeItem(lsSelKey(nodeId, propsKey));
    } else {
      storage.setItem(lsSelKey(nodeId, propsKey), JSON.stringify(list));
    }
  } catch (_e) { /* quota / disabled — non-fatal */ }
}

export function lsClearSel(storage, nodeId, propsKey) {
  try { storage.removeItem(lsSelKey(nodeId, propsKey)); } catch (_e) {}
}
