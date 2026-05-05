import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "PromptLibrary";
const STYLE_ID = "prompt-library-style";

const CSS = `
.pl-gallery { display: flex; flex-direction: column; gap: 6px; padding: 4px; box-sizing: border-box;
  width: 100%; height: 100%; min-height: 0; color: #ddd; font-family: sans-serif; font-size: 12px; }
.pl-toolbar { display: flex; gap: 6px; align-items: center; }
.pl-toolbar input, .pl-toolbar select { flex: 1; min-width: 0; background: #1c1c1c; color: #ddd;
  border: 1px solid #444; padding: 3px 6px; border-radius: 3px; font-size: 12px; }
.pl-toolbar select { flex: 0 0 auto; max-width: 130px; }
.pl-btn { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 3px 8px; cursor: pointer;
  border-radius: 3px; font-size: 12px; }
.pl-btn:hover { background: #383838; }
.pl-grid { flex: 1 1 0; min-height: 0; overflow-y: auto; display: grid; gap: 6px; align-content: start;
  grid-template-columns: repeat(auto-fill, minmax(var(--pl-tile-size, 110px), 1fr));
  grid-auto-rows: max-content;
  padding-right: 2px; }
.pl-tile { position: relative; display: flex; flex-direction: column;
  background: #2a2a2a; border: 2px solid transparent;
  border-radius: 4px; cursor: pointer; overflow: hidden;
  transition: border-color 80ms ease, transform 80ms ease; }
.pl-tile:hover { border-color: #555; transform: scale(1.02); }
.pl-tile.selected, .pl-tile.selected:hover { border-color: #6cf; }
.pl-tile.focused { box-shadow: 0 0 0 2px #f9a inset; }
.pl-tile.dragging { opacity: 0.4; }
.pl-tile.drag-over { outline: 2px dashed #6cf; outline-offset: -4px; }
.pl-tile-img { position: relative; width: 100%; height: 0; padding-bottom: 100%;
  overflow: hidden; background: #1a1a1a; flex: 0 0 auto; }
.pl-tile-img > img, .pl-tile-img > .pl-placeholder {
  position: absolute; inset: 0; }
.pl-tile-check { position: absolute; top: 4px; left: 4px; width: 16px; height: 16px;
  background: rgba(0,0,0,0.7); color: #fff; border: 1px solid #888; border-radius: 3px;
  display: none; align-items: center; justify-content: center; font-size: 11px;
  z-index: 1; cursor: pointer; user-select: none; }
.pl-tile:hover .pl-tile-check, .pl-tile.selected .pl-tile-check { display: flex; }
.pl-tile.selected .pl-tile-check { background: #6cf; color: #111; border-color: #6cf; }
.pl-context-menu { position: fixed; z-index: 10001; background: #2a2a2a; color: #ddd;
  border: 1px solid #444; border-radius: 4px; box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  padding: 4px 0; min-width: 140px; font-size: 12px; user-select: none; }
.pl-context-menu .item { padding: 6px 12px; cursor: pointer; }
.pl-context-menu .item:hover { background: #3a3a3a; }
.pl-context-menu .item.danger { color: #f88; }
.pl-context-menu .sep { height: 1px; background: #444; margin: 4px 0; }
.pl-bulk-bar { display: flex; align-items: center; gap: 6px; padding: 6px 8px;
  background: #1f3550; color: #ddd; border-radius: 4px; font-size: 12px; }
.pl-bulk-bar .count { font-weight: bold; flex: 1; }
.pl-empty-state { grid-column: 1 / -1; padding: 24px 12px; text-align: center;
  color: #888; font-size: 12px; line-height: 1.5; background: #232323;
  border: 1px dashed #444; border-radius: 4px; }
.pl-empty-state strong { color: #ddd; display: block; margin-bottom: 4px; font-size: 13px; }
.pl-search-wrap { position: relative; flex: 1; min-width: 0; display: flex; }
.pl-search-wrap input { width: 100%; padding-right: 22px; }
.pl-search-clear { position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  background: transparent; border: none; color: #888; font-size: 14px;
  cursor: pointer; padding: 0 4px; line-height: 1; }
.pl-search-clear:hover { color: #fff; }
.pl-tile-size { display: flex; align-items: center; gap: 4px; }
.pl-tile-size input { width: 70px; }
.pl-tags-row { display: flex; flex-direction: column; gap: 3px; padding: 0 2px 2px; }
.pl-tag-group { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.pl-tag-group-label { color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  margin-right: 2px; min-width: 60px; }
.pl-tag-chip { background: #2a2a2a; color: #ccc; border: 1px solid #444; padding: 2px 8px;
  border-radius: 10px; font-size: 11px; cursor: pointer; user-select: none; }
.pl-tag-chip:hover { background: #353535; }
.pl-tag-chip.active { background: #2d5070; color: #fff; border-color: #6cf; }
.pl-tag-chip.all { font-weight: bold; }
.pl-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pl-tile .pl-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: #666; }
.pl-tile .pl-name { background: #1c1c1c; color: #ddd; padding: 5px 6px; font-size: 11px;
  line-height: 1.3; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  border-top: 1px solid #111; }
.pl-tile.selected .pl-name { background: #1f3550; color: #fff; }
.pl-add { aspect-ratio: 1 / 1; align-items: center; justify-content: center; font-size: 28px; color: #888;
  background: #232323; border: 2px dashed #555; }
.pl-add:hover { color: #ddd; border-color: #888; }
.pl-grid.list-view { grid-template-columns: 1fr; gap: 4px; grid-auto-rows: max-content; }
.pl-grid.list-view .pl-tile { flex-direction: row; align-items: stretch; min-height: 56px; }
.pl-grid.list-view .pl-tile-img { width: 56px !important; min-width: 56px; height: 56px !important;
  padding-bottom: 0 !important; flex: 0 0 56px !important; }
.pl-grid.list-view .pl-tile .pl-name { flex: 1; display: flex; align-items: center;
  padding: 6px 10px; font-size: 13px; border-top: none; border-left: 1px solid #111; }
.pl-view-toggle { display: flex; gap: 2px; }
.pl-view-toggle .pl-btn { padding: 3px 7px; font-size: 13px; line-height: 1; }
.pl-view-toggle .pl-btn.active { background: #2d5070; border-color: #6cf; color: #fff; }
.pl-modal { position: fixed; z-index: 10000; background: #2a2a2a; color: #ddd; padding: 0 14px 14px;
  border-radius: 6px; width: 460px; max-height: 80vh; overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6); border: 1px solid #444;
  display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; font-size: 13px; }
.pl-modal-header { position: sticky; top: 0; z-index: 1; }
.pl-modal-header { display: flex; align-items: center; gap: 8px; cursor: move;
  user-select: none; padding: 6px 10px; margin: 0 -14px 4px; background: #1f1f1f;
  border-radius: 6px 6px 0 0; border-bottom: 1px solid #444; }
.pl-modal-header h3 { flex: 1; margin: 0; font-size: 13px; }
.pl-modal-close { background: transparent; border: none; color: #aaa; font-size: 18px;
  line-height: 1; cursor: pointer; padding: 0 4px; }
.pl-modal-close:hover { color: #fff; }
.pl-modal label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: #aaa; }
.pl-modal input[type=text], .pl-modal textarea { background: #1c1c1c; color: #ddd; border: 1px solid #444;
  padding: 6px; border-radius: 3px; font-size: 12px; font-family: inherit; }
.pl-modal textarea { resize: vertical; min-height: 100px; }
.pl-modal-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 4px; }
.pl-modal-actions .danger { color: #f88; border-color: #844; }
.pl-thumb-preview { max-width: 120px; max-height: 120px; object-fit: contain;
  background: #1c1c1c; border: 1px solid #444; border-radius: 3px; display: block; }
.pl-status { font-size: 11px; color: #888; min-height: 14px; }
.pl-status.error { color: #f88; }
.pl-history { display: flex; flex-direction: column; gap: 6px; max-height: 240px;
  overflow-y: auto; padding: 4px; background: #1c1c1c; border-radius: 4px;
  margin-top: 4px; }
.pl-history-empty { color: #666; font-size: 11px; padding: 6px; text-align: center; }
.pl-history-row { display: grid; grid-template-columns: auto 1fr auto; gap: 6px;
  align-items: start; padding: 6px; background: #2a2a2a; border-radius: 3px;
  font-size: 11px; }
.pl-history-ts { color: #888; white-space: nowrap; }
.pl-history-body { color: #ccc; word-break: break-word; min-width: 0; }
.pl-history-body strong { color: #fff; display: block; margin-bottom: 2px; }
.pl-history-tags { color: #6cf; font-size: 10px; margin-top: 2px; }
.pl-history-row button { font-size: 10px; padding: 2px 6px; }
`;

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement("style");
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.appendChild(el);
}

async function fetchList() {
  const res = await api.fetchApi("/prompt_library/list");
  const data = await res.json();
  return data.prompts || [];
}

function slugify(name) {
  return (name || "").trim().toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

async function upsert({ id, name, text, tags, imageFile, clearImage }) {
  const body = new FormData();
  if (id) body.append("id", id);
  body.append("name", name);
  body.append("text", text);
  if (tags !== undefined) body.append("tags", tags);
  if (clearImage) body.append("clear_image", "1");
  if (imageFile) body.append("image", imageFile, imageFile.name);
  const res = await api.fetchApi("/prompt_library/upsert", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function deletePrompt(id) {
  const res = await api.fetchApi("/prompt_library/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function importCsv(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  const res = await api.fetchApi("/prompt_library/import_csv", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function importZip(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  const res = await api.fetchApi("/prompt_library/import_zip", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function exportZip(ids) {
  const res = await api.fetchApi("/prompt_library/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: ids || [] }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const count = parseInt(res.headers.get("X-GrimmRibbity-Count") || res.headers.get("X-Ribbity-Count") || "0", 10);
  const blob = await res.blob();
  return { blob, count };
}

async function bulkDelete(ids) {
  const res = await api.fetchApi("/prompt_library/bulk_delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function duplicatePrompt(id, name) {
  const res = await api.fetchApi("/prompt_library/duplicate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function reorderPrompts(ids) {
  const res = await api.fetchApi("/prompt_library/reorder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

let _activeContextMenu = null;
function openContextMenu(x, y, items) {
  if (_activeContextMenu) _activeContextMenu.remove();
  const menu = document.createElement("div");
  menu.className = "pl-context-menu";
  for (const entry of items) {
    if (entry === "sep") {
      const sep = document.createElement("div");
      sep.className = "sep";
      menu.appendChild(sep);
      continue;
    }
    const el = document.createElement("div");
    el.className = "item" + (entry.danger ? " danger" : "");
    el.textContent = entry.label;
    el.onclick = () => {
      menu.remove();
      _activeContextMenu = null;
      entry.action();
    };
    menu.appendChild(el);
  }
  // Position; clamp to viewport.
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  const left = Math.min(x, window.innerWidth - rect.width - 4);
  const top = Math.min(y, window.innerHeight - rect.height - 4);
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  _activeContextMenu = menu;
  const dismiss = (e) => {
    if (!menu.contains(e.target)) {
      menu.remove();
      _activeContextMenu = null;
      document.removeEventListener("mousedown", dismiss, true);
    }
  };
  setTimeout(() => document.addEventListener("mousedown", dismiss, true), 0);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

async function fetchHistory(id) {
  const res = await api.fetchApi(`/prompt_library/history/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function revertPrompt(id, ts) {
  const res = await api.fetchApi("/prompt_library/revert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, ts }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function relativeTime(ts) {
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

const SORT_MODES = {
  manual:      { label: "Manual",     cmp: (a, b) => (a.order||0) - (b.order||0) },
  name_asc:    { label: "Name A-Z",   cmp: (a, b) => a.name.localeCompare(b.name) },
  name_desc:   { label: "Name Z-A",   cmp: (a, b) => b.name.localeCompare(a.name) },
  newest:      { label: "Newest",     cmp: (a, b) => (b.created_at||0) - (a.created_at||0) },
  oldest:      { label: "Oldest",     cmp: (a, b) => (a.created_at||0) - (b.created_at||0) },
  recent_edit: { label: "Recent edit",cmp: (a, b) => (b.updated_at||0) - (a.updated_at||0) },
};
const SORT_KEY = "comfy.PromptLibrary.sort";

// Cache-buster shared across the page. Bumped on websocket update events;
// keeps the same value across re-renders so the browser can cache thumbnails
// instead of re-downloading them every time the gallery refreshes.
let _imageCacheKey = Date.now();
function imageUrl(id) {
  const path = `/prompt_library/image/${id}?v=${_imageCacheKey}`;
  return api.apiURL ? api.apiURL(path) : path;
}

let _modalStack = 0;

function openPromptModal({ existing, onSave, onDelete }) {
  const modal = document.createElement("div");
  modal.className = "pl-modal";

  const header = document.createElement("div");
  header.className = "pl-modal-header";
  const title = document.createElement("h3");
  title.textContent = existing ? "Edit prompt" : "Add prompt";
  const closeX = document.createElement("button");
  closeX.className = "pl-modal-close";
  closeX.type = "button";
  closeX.textContent = "✕";
  closeX.title = "Close";
  header.append(title, closeX);

  const nameLabel = document.createElement("label");
  nameLabel.textContent = "Name";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.value = existing?.name || "";
  nameLabel.appendChild(nameInput);

  const tagsLabel = document.createElement("label");
  tagsLabel.textContent = "Tags (comma-separated; supports category:value)";
  const tagsInput = document.createElement("input");
  tagsInput.type = "text";
  tagsInput.value = (existing?.tags || []).join(", ");
  tagsInput.placeholder = "character, style:cyberpunk, model:anima";
  // Datalist suggests existing tags as the user types — populated lazily.
  const tagsDataList = document.createElement("datalist");
  tagsDataList.id = `pl-tags-${Math.random().toString(36).slice(2, 9)}`;
  tagsInput.setAttribute("list", tagsDataList.id);
  tagsLabel.append(tagsInput, tagsDataList);
  api.fetchApi("/prompt_library/tags").then(r => r.json()).then(d => {
    for (const t of d.tags || []) {
      const opt = document.createElement("option");
      opt.value = t;
      tagsDataList.appendChild(opt);
    }
  }).catch(() => {});

  const idLabel = document.createElement("label");
  idLabel.textContent = existing ? "ID (read-only)" : "ID (optional — auto from name)";
  const idInput = document.createElement("input");
  idInput.type = "text";
  idInput.value = existing?.id || "";
  if (existing) idInput.readOnly = true;
  else nameInput.addEventListener("input", () => {
    if (!idInput.dataset.userEdited) idInput.placeholder = slugify(nameInput.value);
  });
  idInput.addEventListener("input", () => { idInput.dataset.userEdited = "1"; });
  idLabel.appendChild(idInput);

  const textLabel = document.createElement("label");
  textLabel.textContent = "Prompt text";
  const textArea = document.createElement("textarea");
  textArea.value = existing?.text || "";
  textLabel.appendChild(textArea);

  const imgLabel = document.createElement("label");
  imgLabel.textContent = "Reference image (optional)";
  const imgInput = document.createElement("input");
  imgInput.type = "file";
  imgInput.accept = "image/png,image/jpeg,image/webp,image/gif,image/bmp";
  imgLabel.appendChild(imgInput);

  let clearImage = false;
  const preview = document.createElement("img");
  preview.className = "pl-thumb-preview";
  preview.style.display = "none";
  if (existing?.has_image) {
    preview.src = imageUrl(existing.id);
    preview.style.display = "block";
  }
  imgLabel.appendChild(preview);

  if (existing?.has_image) {
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "pl-btn";
    clearBtn.textContent = "Remove image";
    clearBtn.onclick = () => {
      clearImage = true;
      preview.style.display = "none";
      imgInput.value = "";
      clearBtn.disabled = true;
    };
    imgLabel.appendChild(clearBtn);
  }

  // History disclosure (only for existing entries).
  let historyDetails = null;
  if (existing) {
    historyDetails = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "History";
    summary.style.cursor = "pointer";
    summary.style.fontSize = "12px";
    historyDetails.appendChild(summary);

    const list = document.createElement("div");
    list.className = "pl-history";
    list.replaceChildren(Object.assign(document.createElement("div"),
      { className: "pl-history-empty", textContent: "loading..." }));
    historyDetails.appendChild(list);

    const renderHistory = (snapshots) => {
      list.replaceChildren();
      if (!snapshots.length) {
        list.appendChild(Object.assign(document.createElement("div"),
          { className: "pl-history-empty", textContent: "No prior versions yet." }));
        return;
      }
      for (const snap of snapshots) {
        const row = document.createElement("div");
        row.className = "pl-history-row";

        const ts = document.createElement("span");
        ts.className = "pl-history-ts";
        ts.textContent = relativeTime(snap.ts);
        ts.title = new Date(snap.ts * 1000).toLocaleString();

        const body = document.createElement("div");
        body.className = "pl-history-body";
        const nameEl = document.createElement("strong");
        nameEl.textContent = snap.name || "(no name)";
        body.appendChild(nameEl);
        const textEl = document.createElement("div");
        textEl.textContent = (snap.text || "").slice(0, 140) + ((snap.text || "").length > 140 ? "..." : "");
        body.appendChild(textEl);
        if (snap.tags?.length) {
          const tagsEl = document.createElement("div");
          tagsEl.className = "pl-history-tags";
          tagsEl.textContent = snap.tags.join(", ");
          body.appendChild(tagsEl);
        }

        const revertBtn = document.createElement("button");
        revertBtn.type = "button";
        revertBtn.className = "pl-btn";
        revertBtn.textContent = "Revert";
        revertBtn.onclick = async () => {
          if (!confirm(`Revert "${existing.name}" to this version?\n(The current values will be saved to history first.)`)) return;
          revertBtn.disabled = true;
          try {
            await revertPrompt(existing.id, snap.ts);
            close();  // refresh-via-websocket will update the gallery
          } catch (e) {
            revertBtn.disabled = false;
            status.classList.add("error");
            status.textContent = e.message;
          }
        };

        row.append(ts, body, revertBtn);
        list.appendChild(row);
      }
    };

    historyDetails.addEventListener("toggle", async () => {
      if (!historyDetails.open) return;
      try {
        const data = await fetchHistory(existing.id);
        renderHistory(data.history || []);
      } catch (e) {
        list.replaceChildren(Object.assign(document.createElement("div"),
          { className: "pl-history-empty", textContent: `Failed: ${e.message}` }));
      }
    });
  }

  imgInput.onchange = () => {
    const f = imgInput.files[0];
    if (f) {
      preview.src = URL.createObjectURL(f);
      preview.style.display = "block";
      clearImage = false;
    }
  };

  const status = document.createElement("div");
  status.className = "pl-status";

  const actions = document.createElement("div");
  actions.className = "pl-modal-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "pl-btn";
  saveBtn.textContent = "Save";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "pl-btn";
  cancelBtn.textContent = "Cancel";

  // Stop key events from bubbling to LiteGraph (typing in inputs would otherwise
  // trigger canvas shortcuts like delete-node).
  for (const ev of ["keydown", "keyup", "keypress"]) {
    modal.addEventListener(ev, (e) => e.stopPropagation());
  }

  let dragCleanup = null;
  const onKey = (e) => {
    if (e.key === "Escape" && modal.contains(document.activeElement)) {
      e.stopPropagation();
      close();
    }
  };
  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    dragCleanup?.();
    modal.remove();
    _modalStack = Math.max(0, _modalStack - 1);
  };
  document.addEventListener("keydown", onKey, true);
  closeX.onclick = close;
  cancelBtn.onclick = close;

  saveBtn.onclick = async () => {
    const name = nameInput.value.trim();
    if (!name) { status.textContent = "name required"; status.classList.add("error"); return; }
    const customId = idInput.value.trim();
    if (customId && !/^[A-Za-z0-9_-]{1,64}$/.test(customId)) {
      status.textContent = "id must be A-Z, 0-9, _ or - (max 64)";
      status.classList.add("error");
      return;
    }
    saveBtn.disabled = true;
    status.classList.remove("error");
    status.textContent = "saving...";
    try {
      await onSave({
        id: existing?.id || customId,
        name,
        text: textArea.value,
        tags: tagsInput.value,
        imageFile: imgInput.files[0] || null,
        clearImage,
      });
      close();
    } catch (e) {
      saveBtn.disabled = false;
      status.classList.add("error");
      status.textContent = e.message;
    }
  };

  if (existing && onDelete) {
    const delBtn = document.createElement("button");
    delBtn.className = "pl-btn danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm(`Delete "${existing.name}"?`)) return;
      delBtn.disabled = true;
      try { await onDelete(existing.id); close(); }
      catch (e) { delBtn.disabled = false; status.classList.add("error"); status.textContent = e.message; }
    };
    actions.appendChild(delBtn);
    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    actions.appendChild(spacer);
  }
  actions.appendChild(cancelBtn);
  actions.appendChild(saveBtn);

  const children = [header, nameLabel, idLabel, tagsLabel, textLabel, imgLabel];
  if (historyDetails) children.push(historyDetails);
  children.push(status, actions);
  modal.append(...children);

  // Initial position: cascade modals so stacked windows don't overlap exactly.
  const offset = (_modalStack++ % 6) * 24;
  modal.style.left = `calc(50% - 230px + ${offset}px)`;
  modal.style.top = `calc(15% + ${offset}px)`;
  document.body.appendChild(modal);

  dragCleanup = makeDraggable(modal, header);
  nameInput.focus();
}

function makeDraggable(panel, handle) {
  let dragging = false;
  let startX = 0, startY = 0, panelX = 0, panelY = 0;

  const onDown = (e) => {
    if (e.button !== 0 || e.target.closest("button, input, textarea, select")) return;
    const rect = panel.getBoundingClientRect();
    panel.style.left = rect.left + "px";
    panel.style.top = rect.top + "px";
    panelX = rect.left;
    panelY = rect.top;
    startX = e.clientX;
    startY = e.clientY;
    dragging = true;
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!dragging) return;
    const x = panelX + (e.clientX - startX);
    const y = panelY + (e.clientY - startY);
    const maxX = window.innerWidth - panel.offsetWidth;
    const maxY = window.innerHeight - 40;
    panel.style.left = Math.max(0, Math.min(maxX, x)) + "px";
    panel.style.top = Math.max(0, Math.min(maxY, y)) + "px";
  };
  const onUp = () => { dragging = false; };

  handle.addEventListener("mousedown", onDown);
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);

  return () => {
    handle.removeEventListener("mousedown", onDown);
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
}

function buildGallery(node, idWidget) {
  const container = document.createElement("div");
  container.className = "pl-gallery";

  const toolbar = document.createElement("div");
  toolbar.className = "pl-toolbar";

  const searchWrap = document.createElement("div");
  searchWrap.className = "pl-search-wrap";
  const filter = document.createElement("input");
  filter.type = "text";
  filter.placeholder = "search name, text, tags...";
  for (const ev of ["keydown", "keyup", "keypress"]) {
    filter.addEventListener(ev, (e) => e.stopPropagation());
  }
  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "pl-search-clear";
  clearBtn.textContent = "×";
  clearBtn.title = "Clear search";
  clearBtn.style.display = "none";
  clearBtn.onclick = () => { filter.value = ""; clearBtn.style.display = "none"; render(); };
  filter.addEventListener("input", () => {
    clearBtn.style.display = filter.value ? "block" : "none";
  });
  searchWrap.append(filter, clearBtn);
  const modelSelect = document.createElement("select");
  modelSelect.title = "Filter by model (model:* tags)";
  // populated in render() once we know the data

  const sortSelect = document.createElement("select");
  for (const [key, mode] of Object.entries(SORT_MODES)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = mode.label;
    sortSelect.appendChild(opt);
  }
  sortSelect.value = localStorage.getItem(SORT_KEY) || "name_asc";
  sortSelect.title = "Sort";

  const importBtn = document.createElement("button");
  importBtn.className = "pl-btn";
  importBtn.textContent = "Import";
  importBtn.title = "Import from a CSV (name,text,tags,id) or a GrimmRibbity .zip";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".csv,text/csv,.zip,application/zip";
  fileInput.style.display = "none";
  importBtn.onclick = () => fileInput.click();

  const exportBtn = document.createElement("button");
  exportBtn.className = "pl-btn";
  exportBtn.textContent = "Export";
  exportBtn.title = "Export the currently visible prompts (with thumbnails) as a .zip";

  const refreshBtn = document.createElement("button");
  refreshBtn.className = "pl-btn";
  refreshBtn.textContent = "Refresh";

  const SIZE_KEY = "comfy.PromptLibrary.tileSize";
  const sizeWrap = document.createElement("div");
  sizeWrap.className = "pl-tile-size";
  const sizeInput = document.createElement("input");
  sizeInput.type = "range";
  sizeInput.min = "60";
  sizeInput.max = "200";
  sizeInput.step = "8";
  sizeInput.value = localStorage.getItem(SIZE_KEY) || "110";
  sizeInput.title = "Tile size";
  const applySize = () => {
    container.style.setProperty("--pl-tile-size", `${sizeInput.value}px`);
  };
  sizeInput.oninput = () => {
    applySize();
    localStorage.setItem(SIZE_KEY, sizeInput.value);
  };
  sizeWrap.appendChild(sizeInput);

  const VIEW_KEY = "comfy.PromptLibrary.view";
  const viewWrap = document.createElement("div");
  viewWrap.className = "pl-view-toggle";
  const gridViewBtn = document.createElement("button");
  gridViewBtn.className = "pl-btn";
  gridViewBtn.textContent = "▦";
  gridViewBtn.title = "Grid view";
  const listViewBtn = document.createElement("button");
  listViewBtn.className = "pl-btn";
  listViewBtn.textContent = "≡";
  listViewBtn.title = "List view";
  viewWrap.append(gridViewBtn, listViewBtn);
  let viewMode = localStorage.getItem(VIEW_KEY) === "list" ? "list" : "grid";
  const applyView = () => {
    grid.classList.toggle("list-view", viewMode === "list");
    gridViewBtn.classList.toggle("active", viewMode === "grid");
    listViewBtn.classList.toggle("active", viewMode === "list");
    sizeWrap.style.display = viewMode === "list" ? "none" : "";
  };
  gridViewBtn.onclick = () => { viewMode = "grid"; localStorage.setItem(VIEW_KEY, viewMode); applyView(); };
  listViewBtn.onclick = () => { viewMode = "list"; localStorage.setItem(VIEW_KEY, viewMode); applyView(); };

  toolbar.append(searchWrap, modelSelect, sortSelect, sizeWrap, viewWrap, importBtn, exportBtn, refreshBtn, fileInput);

  const tagsRow = document.createElement("div");
  tagsRow.className = "pl-tags-row";

  const grid = document.createElement("div");
  grid.className = "pl-grid";

  let prompts = [];
  let lastVisible = [];
  let focusedIndex = -1;        // for keyboard nav
  const activeTags = new Set();
  // Unified selection: drives both the prompt output (joined into idWidget.value)
  // and bulk actions (Tag/Export/Delete bar).
  const checkedIds = new Set();
  const syncWidget = () => {
    idWidget.value = [...checkedIds].join(",");
    node.setDirtyCanvas(true, true);
  };
  // Repopulate checkedIds from the (comma-separated) widget value. Used on
  // workflow load and on Nodes 2.0 setValue, so the gallery highlights match
  // whatever was saved. Tolerates legacy single-id values.
  const syncFromWidget = () => {
    checkedIds.clear();
    for (const raw of (idWidget.value || "").split(",")) {
      const id = raw.trim();
      if (id) checkedIds.add(id);
    }
  };

  const bulkBar = document.createElement("div");
  bulkBar.className = "pl-bulk-bar";
  bulkBar.style.display = "none";
  const bulkCount = document.createElement("span");
  bulkCount.className = "count";
  const bulkClearBtn = document.createElement("button");
  bulkClearBtn.className = "pl-btn";
  bulkClearBtn.textContent = "Clear";
  bulkClearBtn.onclick = () => { checkedIds.clear(); syncWidget(); render(); };
  const bulkExportBtn = document.createElement("button");
  bulkExportBtn.className = "pl-btn";
  bulkExportBtn.textContent = "Export";
  const bulkTagBtn = document.createElement("button");
  bulkTagBtn.className = "pl-btn";
  bulkTagBtn.textContent = "Tag";
  const bulkDeleteBtn = document.createElement("button");
  bulkDeleteBtn.className = "pl-btn";
  bulkDeleteBtn.style.color = "#f88";
  bulkDeleteBtn.textContent = "Delete";
  bulkBar.append(bulkCount, bulkClearBtn, bulkTagBtn, bulkExportBtn, bulkDeleteBtn);

  container.append(toolbar, tagsRow, bulkBar, grid);

  const updateBulkBar = () => {
    if (checkedIds.size === 0) {
      bulkBar.style.display = "none";
    } else {
      bulkBar.style.display = "flex";
      bulkCount.textContent = `${checkedIds.size} selected`;
    }
  };
  bulkExportBtn.onclick = async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    bulkExportBtn.disabled = true;
    try {
      const { blob, count } = await exportZip(ids);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `ribbity-export-${stamp}-${count}prompts.zip`);
    } catch (e) { alert(`Export failed: ${e.message}`); }
    finally { bulkExportBtn.disabled = false; }
  };
  bulkDeleteBtn.onclick = async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    if (!confirm(`Delete ${ids.length} prompts?`)) return;
    bulkDeleteBtn.disabled = true;
    try {
      await bulkDelete(ids);
      checkedIds.clear();
      syncWidget();
      await refresh();
    } catch (e) { alert(`Delete failed: ${e.message}`); }
    finally { bulkDeleteBtn.disabled = false; }
  };
  bulkTagBtn.onclick = async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    const input = prompt(`Add tags to ${ids.length} prompts (comma-separated). Prefix - to remove (e.g. "-old, new"):`, "");
    if (input === null) return;
    const adds = [], removes = [];
    for (const raw of input.split(",")) {
      const t = raw.trim().toLowerCase();
      if (!t) continue;
      if (t.startsWith("-")) removes.push(t.slice(1).trim());
      else adds.push(t);
    }
    if (!adds.length && !removes.length) return;
    bulkTagBtn.disabled = true;
    try {
      for (const id of ids) {
        const p = prompts.find(x => x.id === id);
        if (!p) continue;
        const newTags = new Set([...(p.tags || []), ...adds]);
        for (const r of removes) newTags.delete(r);
        const fd = new FormData();
        fd.append("id", id);
        fd.append("name", p.name);
        fd.append("text", p.text || "");
        fd.append("tags", [...newTags].join(", "));
        await api.fetchApi("/prompt_library/upsert", { method: "POST", body: fd });
      }
      await refresh();
    } catch (e) { alert(`Tag update failed: ${e.message}`); }
    finally { bulkTagBtn.disabled = false; }
  };

  const updateModelSelect = () => {
    const models = new Set();
    for (const p of prompts) {
      for (const t of p.tags || []) {
        if (t.startsWith("model:")) models.add(t.slice(6));
      }
    }
    const sorted = [...models].sort();
    const previous = modelSelect.value || "";
    modelSelect.replaceChildren();
    const allOpt = document.createElement("option");
    allOpt.value = "";
    allOpt.textContent = sorted.length ? "All models" : "(no model tags)";
    modelSelect.appendChild(allOpt);
    for (const m of sorted) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSelect.appendChild(opt);
    }
    modelSelect.value = sorted.includes(previous) ? previous : "";
    modelSelect.disabled = sorted.length === 0;
  };

  const renderTags = () => {
    // model:* tags are lifted into the modelSelect dropdown — exclude them here.
    // Other tags group by the prefix before ':' (e.g. "style:cyberpunk" -> group "style").
    const groups = new Map();
    for (const p of prompts) {
      for (const t of p.tags || []) {
        if (t.startsWith("model:")) continue;
        const idx = t.indexOf(":");
        const [g, label] = idx > 0 ? [t.slice(0, idx), t.slice(idx + 1)] : ["general", t];
        if (!groups.has(g)) groups.set(g, new Map());
        groups.get(g).set(t, label);
      }
    }

    tagsRow.replaceChildren();

    const topGroup = document.createElement("div");
    topGroup.className = "pl-tag-group";
    const allChip = document.createElement("div");
    allChip.className = "pl-tag-chip all" + (activeTags.size === 0 ? " active" : "");
    allChip.textContent = "All";
    allChip.onclick = () => { activeTags.clear(); render(); };
    topGroup.appendChild(allChip);
    if (groups.size === 0) {
      const hint = document.createElement("span");
      hint.className = "pl-tag-group-label";
      hint.textContent = "no tags yet";
      topGroup.appendChild(hint);
    }
    tagsRow.appendChild(topGroup);

    const sortedGroups = [...groups.keys()].sort((a, b) =>
      a === "general" ? 1 : b === "general" ? -1 : a.localeCompare(b)
    );
    for (const g of sortedGroups) {
      const row = document.createElement("div");
      row.className = "pl-tag-group";
      const lbl = document.createElement("span");
      lbl.className = "pl-tag-group-label";
      lbl.textContent = g === "general" ? "" : g;
      row.appendChild(lbl);
      const tagsInGroup = [...groups.get(g).entries()].sort((a, b) => a[1].localeCompare(b[1]));
      for (const [fullTag, label] of tagsInGroup) {
        const chip = document.createElement("div");
        chip.className = "pl-tag-chip" + (activeTags.has(fullTag) ? " active" : "");
        chip.textContent = label;
        chip.title = fullTag;
        chip.onclick = () => {
          if (activeTags.has(fullTag)) activeTags.delete(fullTag);
          else activeTags.add(fullTag);
          render();
        };
        row.appendChild(chip);
      }
      tagsRow.appendChild(row);
    }
  };

  const render = () => {
    updateModelSelect();
    renderTags();
    grid.replaceChildren();
    const q = filter.value.trim().toLowerCase();
    let visible = prompts;
    const model = modelSelect.value;
    if (model) {
      const want = `model:${model}`;
      visible = visible.filter(p => (p.tags || []).includes(want));
    }
    if (activeTags.size) {
      visible = visible.filter(p => (p.tags || []).some(t => activeTags.has(t)));
    }
    if (q) {
      visible = visible.filter(p => {
        if (p.name.toLowerCase().includes(q)) return true;
        if ((p.text || "").toLowerCase().includes(q)) return true;
        if ((p.tags || []).some(t => t.toLowerCase().includes(q))) return true;
        if ((p.id || "").toLowerCase().includes(q)) return true;
        return false;
      });
    }
    const mode = SORT_MODES[sortSelect.value] || SORT_MODES.name_asc;
    visible = [...visible].sort(mode.cmp);
    // Lift selected tiles to the top so the user can see what's currently
    // checked without scrolling. Stable within each group so the sort order
    // is preserved; deselecting drops the tile back to its natural spot.
    // Skipped in Manual mode because drag-reorder relies on the natural order.
    if (sortSelect.value !== "manual" && checkedIds.size) {
      visible.sort((a, b) => Number(checkedIds.has(b.id)) - Number(checkedIds.has(a.id)));
    }
    lastVisible = visible;
    // Drop checked ids that no longer exist in the library; keep ones that
    // are merely filtered out so multi-select persists across filter changes.
    for (const id of [...checkedIds]) {
      if (!prompts.some(p => p.id === id)) checkedIds.delete(id);
    }
    updateBulkBar();
    if (focusedIndex >= visible.length) focusedIndex = visible.length - 1;

    if (prompts.length === 0) {
      const empty = document.createElement("div");
      empty.className = "pl-empty-state";
      const head = document.createElement("strong");
      head.textContent = "No prompts yet";
      empty.appendChild(head);
      empty.appendChild(document.createTextNode(
        "Click the + tile to add one, or use Import to seed the library from a CSV/ZIP."
      ));
      grid.appendChild(empty);
    } else if (visible.length === 0) {
      const empty = document.createElement("div");
      empty.className = "pl-empty-state";
      const head = document.createElement("strong");
      head.textContent = "No matches";
      empty.appendChild(head);
      empty.appendChild(document.createTextNode(
        "Clear the search/filter or pick a different model."
      ));
      grid.appendChild(empty);
    }
    const isManual = sortSelect.value === "manual";

    visible.forEach((p, idx) => {
      const tile = document.createElement("div");
      tile.className = "pl-tile"
        + (checkedIds.has(p.id) ? " selected" : "")
        + (idx === focusedIndex ? " focused" : "");
      tile.title = p.name;
      tile.dataset.promptId = p.id;
      tile.tabIndex = -1;
      tile.draggable = isManual;

      const tileImg = document.createElement("div");
      tileImg.className = "pl-tile-img";
      if (p.has_image) {
        const img = document.createElement("img");
        img.src = imageUrl(p.id);
        img.loading = "lazy";
        img.alt = p.name;
        img.onerror = () => {
          img.replaceWith(Object.assign(document.createElement("div"),
            { className: "pl-placeholder", textContent: "?" }));
        };
        tileImg.appendChild(img);
      } else {
        const ph = document.createElement("div");
        ph.className = "pl-placeholder";
        ph.textContent = "T";
        tileImg.appendChild(ph);
      }

      const checkbox = document.createElement("div");
      checkbox.className = "pl-tile-check";
      checkbox.textContent = checkedIds.has(p.id) ? "✓" : "";
      checkbox.title = "Toggle selection";
      checkbox.onclick = (e) => {
        e.stopPropagation();
        if (checkedIds.has(p.id)) checkedIds.delete(p.id);
        else checkedIds.add(p.id);
        focusedIndex = idx;
        syncWidget();
        render();
      };
      tileImg.appendChild(checkbox);
      tile.appendChild(tileImg);

      const nm = document.createElement("div");
      nm.className = "pl-name";
      nm.textContent = p.name;
      tile.appendChild(nm);

      tile.onclick = (e) => {
        // Shift-click: range-extend the selection from the focused anchor to here.
        if (e.shiftKey && lastVisible.length) {
          const anchor = focusedIndex >= 0 ? focusedIndex : idx;
          const i0 = Math.min(anchor, idx);
          const i1 = Math.max(anchor, idx);
          for (let i = i0; i <= i1; i++) checkedIds.add(lastVisible[i].id);
          focusedIndex = idx;
          syncWidget();
          render();
          return;
        }
        // Plain or Ctrl/Cmd click: toggle this tile's selection.
        if (checkedIds.has(p.id)) checkedIds.delete(p.id);
        else checkedIds.add(p.id);
        focusedIndex = idx;
        syncWidget();
        render();
      };

      tile.oncontextmenu = (e) => {
        e.preventDefault();
        openContextMenu(e.clientX, e.clientY, [
          { label: "Edit...", action: () => openPromptModal({
              existing: p,
              onSave: async (payload) => { await upsert(payload); await refresh(); },
              onDelete: async (id) => {
                await deletePrompt(id);
                if (checkedIds.delete(id)) syncWidget();
                await refresh();
              },
            }) },
          { label: "Duplicate", action: async () => {
              try { await duplicatePrompt(p.id); await refresh(); }
              catch (err) { alert(`Duplicate failed: ${err.message}`); }
            } },
          { label: "Export this", action: async () => {
              try {
                const { blob } = await exportZip([p.id]);
                const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
                downloadBlob(blob, `ribbity-${p.id}-${stamp}.zip`);
              } catch (err) { alert(`Export failed: ${err.message}`); }
            } },
          "sep",
          { label: "Delete", danger: true, action: async () => {
              if (!confirm(`Delete "${p.name}"?`)) return;
              try {
                await deletePrompt(p.id);
                if (checkedIds.delete(p.id)) syncWidget();
                await refresh();
              } catch (err) { alert(`Delete failed: ${err.message}`); }
            } },
        ]);
      };

      // Drag-and-drop reorder (Manual sort mode only).
      if (isManual) {
        tile.ondragstart = (e) => {
          tile.classList.add("dragging");
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", p.id);
        };
        tile.ondragend = () => tile.classList.remove("dragging");
        tile.ondragover = (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; tile.classList.add("drag-over"); };
        tile.ondragleave = () => tile.classList.remove("drag-over");
        tile.ondrop = async (e) => {
          e.preventDefault();
          tile.classList.remove("drag-over");
          const draggedId = e.dataTransfer.getData("text/plain");
          if (!draggedId || draggedId === p.id) return;
          const ids = visible.map(v => v.id);
          const from = ids.indexOf(draggedId);
          const to = ids.indexOf(p.id);
          if (from < 0 || to < 0) return;
          ids.splice(to, 0, ids.splice(from, 1)[0]);
          try {
            await reorderPrompts(ids);
            await refresh();
          } catch (err) { alert(`Reorder failed: ${err.message}`); }
        };
      }

      grid.appendChild(tile);
    });

    const addTile = document.createElement("div");
    addTile.className = "pl-tile pl-add";
    addTile.textContent = "+";
    addTile.title = "Add prompt";
    addTile.onclick = () => {
      openPromptModal({
        onSave: async (payload) => {
          const created = await upsert(payload);
          checkedIds.add(created.id);
          syncWidget();
          await refresh();
        },
      });
    };
    grid.appendChild(addTile);
  };

  const refresh = async () => {
    try {
      prompts = await fetchList();
      render();
    } catch (e) {
      grid.replaceChildren();
      const err = document.createElement("div");
      err.className = "pl-status error";
      err.textContent = `failed to load: ${e.message}`;
      grid.appendChild(err);
    }
  };

  filter.addEventListener("input", render);
  refreshBtn.onclick = refresh;
  applySize();
  applyView();
  sortSelect.onchange = () => {
    localStorage.setItem(SORT_KEY, sortSelect.value);
    render();
  };
  modelSelect.onchange = render;
  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    if (!file) return;
    const isZip = file.name.toLowerCase().endsWith(".zip") || file.type === "application/zip";
    try {
      const result = isZip ? await importZip(file) : await importCsv(file);
      const kind = isZip ? "ZIP" : "CSV";
      const msg = `${kind} import: ${result.added} added, ${result.updated} updated`
        + (result.errors?.length ? ` (${result.errors.length} errors — see console)` : "");
      if (result.errors?.length) console.warn(`[PromptLibrary] ${kind} import errors:`, result.errors);
      grid.replaceChildren(Object.assign(document.createElement("div"),
        { className: "pl-status", textContent: msg }));
      await refresh();
    } catch (e) {
      grid.replaceChildren(Object.assign(document.createElement("div"),
        { className: "pl-status error", textContent: `Import failed: ${e.message}` }));
    } finally {
      fileInput.value = "";
    }
  };

  exportBtn.onclick = async () => {
    if (!lastVisible.length) {
      alert("Nothing to export (the current filter shows no prompts).");
      return;
    }
    const exportingAll = lastVisible.length === prompts.length;
    const summary = exportingAll
      ? `Export all ${prompts.length} prompts (with thumbnails)?`
      : `Export ${lastVisible.length} of ${prompts.length} visible prompts (with thumbnails)?`;
    if (!confirm(summary)) return;
    exportBtn.disabled = true;
    try {
      const ids = exportingAll ? [] : lastVisible.map(p => p.id);
      const { blob, count } = await exportZip(ids);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `ribbity-export-${stamp}-${count}prompts.zip`);
    } catch (e) {
      alert(`Export failed: ${e.message}`);
    } finally {
      exportBtn.disabled = false;
    }
  };

  // Refresh whenever any save/delete fires server-side (incl. the Save node).
  const onExternal = () => refresh();
  window.addEventListener("prompt-library-updated", onExternal);

  // Make the grid focusable so keyboard nav has somewhere to land.
  grid.tabIndex = 0;
  grid.style.outline = "none";

  const tilesPerRow = () => {
    const tile = grid.querySelector(".pl-tile:not(.pl-add)");
    if (!tile) return 1;
    return Math.max(1, Math.floor(grid.clientWidth / tile.offsetWidth));
  };

  const moveFocus = (delta) => {
    if (!lastVisible.length) return;
    if (focusedIndex < 0) focusedIndex = 0;
    else focusedIndex = Math.max(0, Math.min(lastVisible.length - 1, focusedIndex + delta));
    render();
    const target = grid.querySelectorAll(".pl-tile")[focusedIndex];
    target?.scrollIntoView({ block: "nearest" });
  };

  const onGridKey = (e) => {
    // Slash focuses search from anywhere within the gallery.
    if (e.key === "/" && document.activeElement !== filter) {
      e.preventDefault();
      filter.focus();
      filter.select();
      return;
    }
    // Other keys only fire when grid is focused (not search/etc).
    if (document.activeElement !== grid) return;
    if (e.key === "ArrowRight") { e.preventDefault(); moveFocus(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); moveFocus(-1); }
    else if (e.key === "ArrowDown") { e.preventDefault(); moveFocus(tilesPerRow()); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveFocus(-tilesPerRow()); }
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const p = lastVisible[focusedIndex];
      if (!p) return;
      if (checkedIds.has(p.id)) checkedIds.delete(p.id);
      else checkedIds.add(p.id);
      syncWidget();
      render();
    }
    else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      const p = lastVisible[focusedIndex];
      if (!p) return;
      if (!confirm(`Delete "${p.name}"?`)) return;
      deletePrompt(p.id).then(() => {
        if (checkedIds.delete(p.id)) syncWidget();
        refresh();
      }).catch(err => alert(`Delete failed: ${err.message}`));
    }
    else if (e.key === "Escape") {
      e.preventDefault();
      checkedIds.clear();
      syncWidget();
      focusedIndex = -1;
      render();
    }
  };
  container.addEventListener("keydown", onGridKey);
  // Stop key events from propagating to LiteGraph when interacting with the gallery.
  container.addEventListener("keydown", (e) => e.stopPropagation());
  // Let the wheel scroll the grid (and other inner scrollers) instead of
  // zooming the LiteGraph canvas. Capture phase + manual scroll so we beat
  // LiteGraph's wheel handler, which otherwise eats the event for canvas
  // zoom even though the cursor is over our DOM widget.
  container.addEventListener("wheel", (e) => {
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16;          // lines → px
    else if (e.deltaMode === 2) dy *= e.target?.clientHeight || 400;  // pages → px
    for (let el = e.target; el && el !== container.parentNode; el = el.parentNode) {
      if (!(el instanceof HTMLElement)) continue;
      const style = getComputedStyle(el);
      if (!/(auto|scroll)/.test(style.overflowY)) continue;
      if (el.scrollHeight <= el.clientHeight) continue;
      const atTop = el.scrollTop <= 0;
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
      if ((dy < 0 && atTop) || (dy > 0 && atBottom)) return;  // let canvas zoom at edges
      el.scrollTop += dy;
      e.preventDefault();
      e.stopPropagation();
      return;
    }
  }, { passive: false, capture: true });

  container._promptLibraryCleanup = () => {
    window.removeEventListener("prompt-library-updated", onExternal);
  };

  grid.replaceChildren(Object.assign(document.createElement("div"), {
    className: "pl-status", textContent: "loading...",
  }));
  syncFromWidget();
  refresh();

  return { container, refresh, render, syncFromWidget };
}

let _wsListenerInstalled = false;
function installWebsocketBridge() {
  if (_wsListenerInstalled) return;
  _wsListenerInstalled = true;
  api.addEventListener("prompt_library.updated", () => {
    _imageCacheKey = Date.now();  // bust thumbnail cache on any change
    window.dispatchEvent(new CustomEvent("prompt-library-updated"));
  });
}

app.registerExtension({
  name: "comfy.PromptLibrary",
  async setup() {
    installWebsocketBridge();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    injectStyle();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const idWidget = this.widgets.find(w => w.name === "prompt_id");
      if (idWidget) {
        // Hide the underlying string widget; gallery clicks write to its value,
        // and ComfyUI auto-serializes it into the workflow JSON. We hide via
        // three mechanisms to cover both the legacy LiteGraph renderer and
        // the new Nodes 2.0 renderer:
        //   - widget.hidden = true       → Nodes 2.0 (Vue-rendered) skips it
        //   - computeSize → [0, -4]      → legacy renderer collapses the row
        //   - draw = noop                → legacy renderer extra safety
        // Do NOT mutate widget.type to a custom string — Nodes 2.0's typed
        // renderer treats unknown types as broken and bails on the whole node.
        idWidget.hidden = true;
        idWidget.computeSize = () => [0, -4];
        idWidget.draw = () => {};
      }

      const { container, render, syncFromWidget } = buildGallery(this, idWidget);
      // Defensive: container needs explicit dimensions because Nodes 2.0
      // doesn't always give DOM widgets a sized wrapper before first paint.
      container.style.minHeight = "240px";
      container.style.width = "100%";

      const galleryWidget = this.addDOMWidget("gallery", "PromptLibraryGallery", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 240,
        // Nodes 2.0 may call getValue/setValue during reactivity sync; without
        // these the widget can be treated as malformed and skipped.
        getValue: () => idWidget?.value || "",
        setValue: (v) => {
          if (idWidget) idWidget.value = v;
          syncFromWidget?.();
          render?.();
        },
      });
      this._promptLibraryRender = render;
      this._promptLibrarySyncFromWidget = syncFromWidget;
      this._promptLibraryGalleryWidget = galleryWidget;

      this.size = [320, 320];
      if (typeof this.setSize === "function") this.setSize([320, 320]);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // Re-render so the tile matching the workflow's saved prompt_id gets highlighted.
      this._promptLibrarySyncFromWidget?.();
      this._promptLibraryRender?.();
      return r;
    };
  },
});
