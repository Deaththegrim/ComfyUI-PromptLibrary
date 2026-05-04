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
.pl-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pl-tile .pl-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: #666; }
.pl-tile .pl-name { position: absolute; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.65);
  color: #fff; padding: 2px 4px; font-size: 10px; text-align: center;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pl-add { display: flex; align-items: center; justify-content: center; font-size: 28px; color: #888;
  background: #232323; border: 2px dashed #555; }
.pl-add:hover { color: #ddd; border-color: #888; }
.pl-modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 10000;
  display: flex; align-items: center; justify-content: center; }
.pl-modal { background: #2a2a2a; color: #ddd; padding: 16px; border-radius: 6px; min-width: 380px;
  max-width: 520px; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; font-size: 13px; }
.pl-modal h3 { margin: 0; font-size: 14px; }
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

async function upsert({ id, name, text, imageFile, clearImage }) {
  const body = new FormData();
  if (id) body.append("id", id);
  body.append("name", name);
  body.append("text", text);
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

function openPromptModal({ existing, onSave, onDelete }) {
  const bg = document.createElement("div");
  bg.className = "pl-modal-bg";
  const modal = document.createElement("div");
  modal.className = "pl-modal";

  const title = document.createElement("h3");
  title.textContent = existing ? "Edit prompt" : "Add prompt";

  const nameLabel = document.createElement("label");
  nameLabel.textContent = "Name";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.value = existing?.name || "";
  nameLabel.appendChild(nameInput);

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

  const onKey = (e) => {
    if (e.key === "Escape") { e.stopPropagation(); close(); }
  };
  const close = () => { document.removeEventListener("keydown", onKey, true); bg.remove(); };
  document.addEventListener("keydown", onKey, true);
  cancelBtn.onclick = close;
  bg.onclick = (e) => { if (e.target === bg) close(); };

  saveBtn.onclick = async () => {
    const name = nameInput.value.trim();
    if (!name) { status.textContent = "name required"; status.classList.add("error"); return; }
    saveBtn.disabled = true;
    status.classList.remove("error");
    status.textContent = "saving...";
    try {
      await onSave({
        id: existing?.id,
        name,
        text: textArea.value,
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

  modal.append(title, nameLabel, textLabel, imgLabel, status, actions);
  bg.appendChild(modal);
  document.body.appendChild(bg);
  nameInput.focus();
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

  const grid = document.createElement("div");
  grid.className = "pl-grid";

  container.append(toolbar, grid);

  let prompts = [];

  const render = () => {
    grid.replaceChildren();
    const q = filter.value.trim().toLowerCase();
    const visible = q ? prompts.filter(p => p.name.toLowerCase().includes(q)) : prompts;
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

  grid.replaceChildren(Object.assign(document.createElement("div"), {
    className: "pl-status", textContent: "loading...",
  }));
  refresh();

  return { container, refresh, render };
}

app.registerExtension({
  name: "comfy.PromptLibrary",
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
