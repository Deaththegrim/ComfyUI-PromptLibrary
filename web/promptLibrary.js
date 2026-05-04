import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "PromptLibrary";
const STYLE_ID = "prompt-library-style";

const CSS = `
.pl-gallery { display: flex; flex-direction: column; gap: 6px; padding: 4px; box-sizing: border-box;
  width: 100%; height: 100%; min-height: 0; color: #ddd; font-family: sans-serif; font-size: 12px; }
.pl-toolbar { display: flex; gap: 6px; align-items: center; }
.pl-toolbar input { flex: 1; min-width: 0; background: #1c1c1c; color: #ddd; border: 1px solid #444;
  padding: 3px 6px; border-radius: 3px; font-size: 12px; }
.pl-btn { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 3px 8px; cursor: pointer;
  border-radius: 3px; font-size: 12px; }
.pl-btn:hover { background: #383838; }
.pl-grid { flex: 1; overflow-y: auto; display: grid; gap: 6px; align-content: start;
  grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); padding-right: 2px; }
.pl-tile { position: relative; aspect-ratio: 1 / 1; background: #2a2a2a; border: 2px solid transparent;
  border-radius: 4px; cursor: pointer; overflow: hidden; }
.pl-tile.selected { border-color: #6cf; }
.pl-tags-row { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 2px 2px; }
.pl-tag-chip { background: #2a2a2a; color: #ccc; border: 1px solid #444; padding: 2px 8px;
  border-radius: 10px; font-size: 11px; cursor: pointer; user-select: none; }
.pl-tag-chip:hover { background: #353535; }
.pl-tag-chip.active { background: #2d5070; color: #fff; border-color: #6cf; }
.pl-tag-chip.all { font-weight: bold; }
.pl-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pl-tile .pl-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: #666; }
.pl-tile .pl-name { position: absolute; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.65);
  color: #fff; padding: 2px 4px; font-size: 10px; text-align: center;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pl-add { display: flex; align-items: center; justify-content: center; font-size: 28px; color: #888;
  background: #232323; border: 2px dashed #555; }
.pl-add:hover { color: #ddd; border-color: #888; }
.pl-modal { position: fixed; z-index: 10000; background: #2a2a2a; color: #ddd; padding: 0 14px 14px;
  border-radius: 6px; width: 460px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); border: 1px solid #444;
  display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; font-size: 13px; }
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

function imageUrl(id) {
  return api.apiURL ? api.apiURL(`/prompt_library/image/${id}?t=${Date.now()}`)
                    : `/prompt_library/image/${id}?t=${Date.now()}`;
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
  tagsLabel.textContent = "Tags (comma-separated)";
  const tagsInput = document.createElement("input");
  tagsInput.type = "text";
  tagsInput.value = (existing?.tags || []).join(", ");
  tagsInput.placeholder = "character, fantasy, sci-fi";
  tagsLabel.appendChild(tagsInput);

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

  modal.append(header, nameLabel, idLabel, tagsLabel, textLabel, imgLabel, status, actions);

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
  const filter = document.createElement("input");
  filter.type = "text";
  filter.placeholder = "filter...";
  // Stop key events bubbling to LiteGraph (otherwise typing triggers canvas shortcuts).
  for (const ev of ["keydown", "keyup", "keypress"]) {
    filter.addEventListener(ev, (e) => e.stopPropagation());
  }
  const refreshBtn = document.createElement("button");
  refreshBtn.className = "pl-btn";
  refreshBtn.textContent = "Refresh";
  toolbar.append(filter, refreshBtn);

  const tagsRow = document.createElement("div");
  tagsRow.className = "pl-tags-row";

  const grid = document.createElement("div");
  grid.className = "pl-grid";

  container.append(toolbar, tagsRow, grid);

  let prompts = [];
  const activeTags = new Set();

  const renderTags = () => {
    const seen = new Set();
    for (const p of prompts) for (const t of p.tags || []) seen.add(t);
    const all = [...seen].sort();
    tagsRow.replaceChildren();

    const allChip = document.createElement("div");
    allChip.className = "pl-tag-chip all" + (activeTags.size === 0 ? " active" : "");
    allChip.textContent = "All";
    allChip.onclick = () => { activeTags.clear(); render(); };
    tagsRow.appendChild(allChip);

    for (const tag of all) {
      const chip = document.createElement("div");
      chip.className = "pl-tag-chip" + (activeTags.has(tag) ? " active" : "");
      chip.textContent = tag;
      chip.onclick = () => {
        if (activeTags.has(tag)) activeTags.delete(tag);
        else activeTags.add(tag);
        render();
      };
      tagsRow.appendChild(chip);
    }
  };

  const render = () => {
    renderTags();
    grid.replaceChildren();
    const q = filter.value.trim().toLowerCase();
    let visible = prompts;
    if (activeTags.size) {
      visible = visible.filter(p => (p.tags || []).some(t => activeTags.has(t)));
    }
    if (q) visible = visible.filter(p => p.name.toLowerCase().includes(q));
    for (const p of visible) {
      const tile = document.createElement("div");
      tile.className = "pl-tile" + (p.id === idWidget.value ? " selected" : "");
      tile.title = p.name;
      if (p.has_image) {
        const img = document.createElement("img");
        img.src = imageUrl(p.id);
        img.loading = "lazy";
        tile.appendChild(img);
      } else {
        const ph = document.createElement("div");
        ph.className = "pl-placeholder";
        ph.textContent = "T";
        tile.appendChild(ph);
      }
      const nm = document.createElement("div");
      nm.className = "pl-name";
      nm.textContent = p.name;
      tile.appendChild(nm);

      tile.onclick = () => {
        const wasSelected = idWidget.value === p.id;
        idWidget.value = wasSelected ? "" : p.id;
        node.setDirtyCanvas(true, true);
        for (const t of grid.querySelectorAll(".pl-tile")) t.classList.remove("selected");
        if (!wasSelected) tile.classList.add("selected");
      };
      tile.oncontextmenu = (e) => {
        e.preventDefault();
        openPromptModal({
          existing: p,
          onSave: async (payload) => { await upsert(payload); await refresh(); },
          onDelete: async (id) => {
            await deletePrompt(id);
            if (idWidget.value === id) idWidget.value = "";
            await refresh();
          },
        });
      };
      grid.appendChild(tile);
    }

    const addTile = document.createElement("div");
    addTile.className = "pl-tile pl-add";
    addTile.textContent = "+";
    addTile.title = "Add prompt";
    addTile.onclick = () => {
      openPromptModal({
        onSave: async (payload) => {
          const created = await upsert(payload);
          idWidget.value = created.id;
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

  filter.oninput = render;
  refreshBtn.onclick = refresh;

  // Refresh whenever any save/delete fires server-side (incl. the Save node).
  const onExternal = () => refresh();
  window.addEventListener("prompt-library-updated", onExternal);
  container._promptLibraryCleanup = () => {
    window.removeEventListener("prompt-library-updated", onExternal);
  };

  grid.replaceChildren(Object.assign(document.createElement("div"), {
    className: "pl-status", textContent: "loading...",
  }));
  refresh();

  return { container, refresh, render };
}

let _wsListenerInstalled = false;
function installWebsocketBridge() {
  if (_wsListenerInstalled) return;
  _wsListenerInstalled = true;
  api.addEventListener("prompt_library.updated", () => {
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
        // and ComfyUI auto-serializes it into the workflow JSON.
        idWidget.type = "hidden_prompt_id";
        idWidget.computeSize = () => [0, -4];
        idWidget.draw = () => {};
      }

      const { container, render } = buildGallery(this, idWidget);
      this.addDOMWidget("gallery", "PromptLibraryGallery", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 240,
      });
      this._promptLibraryRender = render;

      this.size = [320, 320];
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // Re-render so the tile matching the workflow's saved prompt_id gets highlighted.
      this._promptLibraryRender?.();
      return r;
    };
  },
});
