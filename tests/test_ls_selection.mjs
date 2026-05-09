// Node 18+ built-in test runner. Run from project root:
//   node --test tests/test_ls_selection.mjs
//
// Covers the two ghost-select fixes shipped in v0.57.1:
//   1. Multi-Library panels write to distinct localStorage keys
//   2. cleanup() clears the entry so a new same-id node starts fresh

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  lsSelKey, lsReadSel, lsWriteSel, lsClearSel,
} from "../web/lsSelection.js";

function makeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    _size: () => map.size,
    _keys: () => [...map.keys()],
    _raw: map,
  };
}

test("lsSelKey: includes both nodeId and propsKey", () => {
  assert.equal(lsSelKey(5, "pl_state"), "pl_sel_5_pl_state");
  assert.equal(lsSelKey(5, "pl_state_1"), "pl_sel_5_pl_state_1");
  assert.equal(lsSelKey(5, "pl_state_2"), "pl_sel_5_pl_state_2");
});

test("lsSelKey: tolerates missing nodeId / propsKey", () => {
  assert.equal(lsSelKey(undefined, "pl_state"), "pl_sel_anon_pl_state");
  assert.equal(lsSelKey(null, "pl_state"), "pl_sel_anon_pl_state");
  assert.equal(lsSelKey(5, undefined), "pl_sel_5_pl_state");
  assert.equal(lsSelKey(5, ""), "pl_sel_5_pl_state");
});

test("Multi panels do not collide: writing to one panel leaves others intact", () => {
  const s = makeStorage();
  const nodeId = 7;
  lsWriteSel(s, nodeId, "pl_state_1", ["a", "b"]);
  lsWriteSel(s, nodeId, "pl_state_2", ["c"]);
  lsWriteSel(s, nodeId, "pl_state_3", ["d", "e", "f"]);

  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_1"), ["a", "b"]);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_2"), ["c"]);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_3"), ["d", "e", "f"]);
  assert.equal(s._size(), 3, "exactly three distinct keys, one per panel");
});

test("Multi panels: clearing one panel leaves the other two intact", () => {
  const s = makeStorage();
  const nodeId = 7;
  lsWriteSel(s, nodeId, "pl_state_1", ["a"]);
  lsWriteSel(s, nodeId, "pl_state_2", ["b"]);
  lsWriteSel(s, nodeId, "pl_state_3", ["c"]);

  lsClearSel(s, nodeId, "pl_state_2");

  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_1"), ["a"]);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_2"), []);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_3"), ["c"]);
});

test("cleanup-on-remove: clearing for a deleted node frees the slot for reuse", () => {
  const s = makeStorage();
  // Node A at id=5 selected entry "x".
  lsWriteSel(s, 5, "pl_state", ["x"]);
  assert.deepEqual(lsReadSel(s, 5, "pl_state"), ["x"]);

  // User deletes node A — cleanup() runs and clears.
  lsClearSel(s, 5, "pl_state");

  // Node B added later, Comfy reuses id=5. Read returns empty — no ghost.
  assert.deepEqual(lsReadSel(s, 5, "pl_state"), []);
  assert.equal(s._size(), 0);
});

test("cleanup of one panel does not affect the other panels of the same node", () => {
  const s = makeStorage();
  const nodeId = 9;
  lsWriteSel(s, nodeId, "pl_state_1", ["a"]);
  lsWriteSel(s, nodeId, "pl_state_2", ["b"]);
  lsWriteSel(s, nodeId, "pl_state_3", ["c"]);

  lsClearSel(s, nodeId, "pl_state_1");

  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_1"), []);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_2"), ["b"]);
  assert.deepEqual(lsReadSel(s, nodeId, "pl_state_3"), ["c"]);
});

test("write with empty array removes the entry instead of storing []", () => {
  const s = makeStorage();
  lsWriteSel(s, 1, "pl_state", ["a"]);
  assert.equal(s._size(), 1);
  lsWriteSel(s, 1, "pl_state", []);
  assert.equal(s._size(), 0, "empty selection must purge the key");
});

test("read tolerates corrupted JSON without throwing", () => {
  const s = makeStorage();
  s.setItem(lsSelKey(1, "pl_state"), "{not valid json");
  assert.deepEqual(lsReadSel(s, 1, "pl_state"), []);
});

test("read filters out non-string entries (defensive against tampering)", () => {
  const s = makeStorage();
  s.setItem(lsSelKey(1, "pl_state"), JSON.stringify(["a", 42, null, "b", { x: 1 }]));
  assert.deepEqual(lsReadSel(s, 1, "pl_state"), ["a", "b"]);
});

test("read returns [] when payload is not a JSON array", () => {
  const s = makeStorage();
  s.setItem(lsSelKey(1, "pl_state"), JSON.stringify({ "wrong": "shape" }));
  assert.deepEqual(lsReadSel(s, 1, "pl_state"), []);
});

test("write filters non-string ids and skips empties", () => {
  const s = makeStorage();
  lsWriteSel(s, 1, "pl_state", ["a", "", "b", null, 7, "c"]);
  assert.deepEqual(lsReadSel(s, 1, "pl_state"), ["a", "b", "c"]);
});

test("storage failures do not throw out of write/clear", () => {
  const s = {
    getItem: () => { throw new Error("disabled"); },
    setItem: () => { throw new Error("disabled"); },
    removeItem: () => { throw new Error("disabled"); },
  };
  assert.doesNotThrow(() => lsWriteSel(s, 1, "pl_state", ["a"]));
  assert.doesNotThrow(() => lsClearSel(s, 1, "pl_state"));
  assert.deepEqual(lsReadSel(s, 1, "pl_state"), []);
});

test("regression: pre-fix shape collision is impossible — different propsKey yields different key", () => {
  // Pre-v0.57.1 the key was just `pl_sel_${nodeId}` ignoring propsKey.
  // This guard ensures the fix can't silently regress.
  const k1 = lsSelKey(7, "pl_state_1");
  const k2 = lsSelKey(7, "pl_state_2");
  const k3 = lsSelKey(7, "pl_state_3");
  assert.notEqual(k1, k2);
  assert.notEqual(k2, k3);
  assert.notEqual(k1, k3);
});
