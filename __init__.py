import csv
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from aiohttp import web
from server import PromptServer

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
STORE_PATH = DATA_DIR / "prompts.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _load() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[PromptLibrary] failed to read {STORE_PATH}: {e}; treating as empty")
        return []
    return data if isinstance(data, list) else []


_last_known_mtime = 0.0


def _save(items: list[dict]) -> None:
    global _last_known_mtime
    tmp = STORE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    tmp.replace(STORE_PATH)
    try:
        _last_known_mtime = STORE_PATH.stat().st_mtime
    except OSError:
        pass


def _safe_id(value: str) -> str | None:
    return value if value and _ID_RE.match(value) else None


def _slugify(name: str) -> str:
    """Turn 'Cyberpunk Style 2!' into 'cyberpunk_style_2'. May return '' for all-symbol names."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip().lower())
    return s.strip("_")[:64]


def _unique_id(base: str, existing_ids: set[str]) -> str:
    """Append _2, _3, ... until the id is free; fall back to uuid for an empty base."""
    if not base:
        return uuid.uuid4().hex[:12]
    if base not in existing_ids:
        return base
    n = 2
    trunk = base[:60]
    while True:
        candidate = f"{trunk}_{n}"
        if candidate not in existing_ids:
            return candidate
        n += 1


def _parse_tags(value) -> list[str]:
    """Accept a comma-separated string or a list; return cleaned, deduped, lowercased tags."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return []
    seen = []
    for p in parts:
        t = str(p).strip().lower()
        if t and t not in seen:
            seen.append(t)
    return seen


def _image_path_for(prompt_id: str) -> Path | None:
    for ext in _ALLOWED_IMAGE_EXT:
        p = IMAGES_DIR / f"{prompt_id}{ext}"
        if p.exists():
            return p
    return None


def _delete_image_files(prompt_id: str) -> None:
    for ext in _ALLOWED_IMAGE_EXT:
        p = IMAGES_DIR / f"{prompt_id}{ext}"
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def _save_image_tensor(prompt_id: str, image) -> bool:
    """Save the first frame of a ComfyUI IMAGE batch as PNG. Returns True on success."""
    if image is None:
        return False
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"[PromptLibrary] PIL/numpy unavailable, can't save thumbnail: {e}")
        return False
    try:
        frame = image[0]
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        arr = (frame.clip(0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        elif arr.shape[-1] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            pil = Image.fromarray(arr[..., :3], mode="RGB")
        _delete_image_files(prompt_id)
        pil.save(IMAGES_DIR / f"{prompt_id}.png", format="PNG")
        return True
    except Exception as e:
        print(f"[PromptLibrary] failed to save thumbnail for {prompt_id!r}: {e}")
        return False


def _notify_change() -> None:
    """Push a websocket event so any open gallery widgets can refresh themselves."""
    try:
        PromptServer.instance.send_sync("prompt_library.updated", {})
    except Exception:
        pass


def _now() -> float:
    return time.time()


def _touch(item: dict, *, created: bool) -> None:
    now = _now()
    if created or "created_at" not in item:
        item.setdefault("created_at", now)
    item["updated_at"] = now


_HISTORY_CAP = 20


def _push_history(item: dict) -> None:
    """Snapshot the current name/text/tags onto the entry's history list."""
    history = list(item.get("history") or [])
    history.append({
        "ts": _now(),
        "name": item.get("name", ""),
        "text": item.get("text", ""),
        "tags": list(item.get("tags") or []),
    })
    item["history"] = history[-_HISTORY_CAP:]


def _maybe_push_history(item: dict, new_name: str, new_text: str, new_tags: list) -> None:
    """Push history only if any user-visible field actually changes (image excluded)."""
    if (item.get("name", "") == new_name
        and item.get("text", "") == new_text
        and list(item.get("tags") or []) == list(new_tags or [])):
        return
    _push_history(item)


def _start_watcher() -> None:
    """Daemon thread polling prompts.json mtime; pushes refresh on external edit."""
    global _last_known_mtime
    try:
        _last_known_mtime = STORE_PATH.stat().st_mtime if STORE_PATH.exists() else 0.0
    except OSError:
        _last_known_mtime = 0.0

    def loop():
        global _last_known_mtime
        while True:
            time.sleep(2)
            try:
                mtime = STORE_PATH.stat().st_mtime if STORE_PATH.exists() else 0.0
            except OSError:
                continue
            if mtime != _last_known_mtime:
                _last_known_mtime = mtime
                _notify_change()

    t = threading.Thread(target=loop, daemon=True, name="prompt-library-watcher")
    t.start()


_start_watcher()


class PromptLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_id": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, prompt_id):
        with _lock:
            for item in _load():
                if item.get("id") == prompt_id:
                    return item.get("text", "")
        return ""

    def load_prompt(self, prompt_id: str):
        with _lock:
            for item in _load():
                if item.get("id") == prompt_id:
                    return (item.get("text", ""),)
        if prompt_id:
            print(f"[PromptLibrary] no prompt with id={prompt_id!r}; returning empty string")
        return ("",)


class PromptLibrarySave:
    """Save a prompt to the library when the workflow runs.

    Wire any STRING source into `text`, an IMAGE source into `thumbnail` (optional),
    type the name as a widget. On Queue, the entry is appended/updated and the
    open gallery widgets refresh automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "thumbnail": ("IMAGE",),
                "tags": ("STRING", {"default": "", "multiline": False}),
                "prompt_id": ("STRING", {"default": "", "multiline": False}),
                "overwrite_by_name": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "id")
    FUNCTION = "save"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def save(self, name, text, thumbnail=None, tags="", prompt_id="", overwrite_by_name=False):
        name = (name or "").strip()
        if not name:
            raise ValueError("PromptLibrarySave: name is required")
        prompt_id = (prompt_id or "").strip()
        if prompt_id and not _safe_id(prompt_id):
            raise ValueError(f"PromptLibrarySave: invalid prompt_id {prompt_id!r}")
        parsed_tags = _parse_tags(tags)

        with _lock:
            items = _load()
            existing = None
            if prompt_id:
                existing = next((i for i in items if i.get("id") == prompt_id), None)
            elif overwrite_by_name:
                existing = next((i for i in items if i.get("name") == name), None)

            created = existing is None
            if created:
                pid = prompt_id or _unique_id(_slugify(name), {i.get("id") for i in items})
                existing = {"id": pid}
                items.append(existing)
            else:
                _maybe_push_history(existing, name, text or "", parsed_tags)

            existing["name"] = name
            existing["text"] = text or ""
            existing["tags"] = parsed_tags
            _touch(existing, created=created)

            if thumbnail is not None:
                _save_image_tensor(existing["id"], thumbnail)

            _save(items)
            saved_id = existing["id"]

        _notify_change()
        print(f"[PromptLibrary] saved id={saved_id!r} name={name!r} tags={parsed_tags}")
        return (text or "", saved_id)


routes = PromptServer.instance.routes


@routes.get("/prompt_library/list")
async def list_prompts(_request):
    with _lock:
        items = _load()
    out = []
    for item in items:
        pid = item.get("id", "")
        out.append({
            "id": pid,
            "name": item.get("name", ""),
            "text": item.get("text", ""),
            "tags": item.get("tags", []),
            "created_at": item.get("created_at", 0),
            "updated_at": item.get("updated_at", 0),
            "has_image": _image_path_for(pid) is not None,
        })
    return web.json_response({"prompts": out})


@routes.get("/prompt_library/tags")
async def list_tags(_request):
    with _lock:
        items = _load()
    seen = set()
    for item in items:
        for t in item.get("tags", []) or []:
            seen.add(str(t).strip().lower())
    return web.json_response({"tags": sorted(t for t in seen if t)})


@routes.get("/prompt_library/image/{prompt_id}")
async def get_image(request):
    pid = _safe_id(request.match_info.get("prompt_id", ""))
    if not pid:
        return web.Response(status=400, text="invalid id")
    path = _image_path_for(pid)
    if not path:
        return web.Response(status=404)
    return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


@routes.post("/prompt_library/upsert")
async def upsert_prompt(request):
    reader = await request.post()
    pid = (reader.get("id") or "").strip()
    name = (reader.get("name") or "").strip()
    text = reader.get("text") or ""
    tags = _parse_tags(reader.get("tags"))
    clear_image = (reader.get("clear_image") or "") == "1"
    image_field = reader.get("image")

    if not name:
        return web.json_response({"error": "name required"}, status=400)
    if pid and not _safe_id(pid):
        return web.json_response({"error": "invalid id"}, status=400)

    with _lock:
        items = _load()
        if not pid:
            pid = _unique_id(_slugify(name), {i.get("id") for i in items})
        existing = next((i for i in items if i.get("id") == pid), None)
        created = existing is None
        if created:
            existing = {"id": pid}
            items.append(existing)
        else:
            _maybe_push_history(existing, name, text, tags)
        existing["name"] = name
        existing["text"] = text
        existing["tags"] = tags
        _touch(existing, created=created)

        if clear_image:
            _delete_image_files(pid)

        if image_field is not None and hasattr(image_field, "file") and image_field.filename:
            ext = os.path.splitext(image_field.filename)[1].lower()
            if ext not in _ALLOWED_IMAGE_EXT:
                return web.json_response({"error": f"unsupported image type {ext}"}, status=400)
            data = image_field.file.read()
            if len(data) > _MAX_IMAGE_BYTES:
                return web.json_response({"error": "image too large"}, status=400)
            _delete_image_files(pid)
            (IMAGES_DIR / f"{pid}{ext}").write_bytes(data)

        _save(items)

    _notify_change()
    return web.json_response({
        "id": pid,
        "name": name,
        "text": text,
        "tags": tags,
        "has_image": _image_path_for(pid) is not None,
    })


def _import_csv(text: str) -> tuple[int, int, list[str]]:
    """Parse CSV body and upsert each row. Returns (added, updated, errors)."""
    reader = csv.DictReader(io.StringIO(text))
    added = updated = 0
    errors: list[str] = []
    if not reader.fieldnames or "name" not in reader.fieldnames:
        return 0, 0, ["CSV missing required 'name' column"]

    with _lock:
        items = _load()
        index = {i.get("id"): i for i in items}
        for row_num, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                errors.append(f"row {row_num}: empty name")
                continue
            text_val = row.get("text", "") or ""
            # Tags use ';' inside CSV cell since ',' is the field delimiter.
            tags = _parse_tags((row.get("tags") or "").replace(";", ","))
            row_id = (row.get("id") or "").strip()
            if row_id and not _safe_id(row_id):
                errors.append(f"row {row_num}: invalid id {row_id!r}")
                continue
            if not row_id:
                row_id = _unique_id(_slugify(name), set(index.keys()))

            existing = index.get(row_id)
            created = existing is None
            if created:
                existing = {"id": row_id}
                items.append(existing)
                index[row_id] = existing
                added += 1
            else:
                _maybe_push_history(existing, name, text_val, tags)
                updated += 1
            existing["name"] = name
            existing["text"] = text_val
            existing["tags"] = tags
            _touch(existing, created=created)

        _save(items)

    return added, updated, errors


@routes.post("/prompt_library/import_csv")
async def import_csv_route(request):
    reader = await request.post()
    field = reader.get("file")
    if field is not None and hasattr(field, "file"):
        body = field.file.read().decode("utf-8", errors="replace")
    else:
        body = reader.get("csv") or ""
    if not body.strip():
        return web.json_response({"error": "no CSV body provided"}, status=400)
    added, updated, errors = _import_csv(body)
    _notify_change()
    return web.json_response({"added": added, "updated": updated, "errors": errors})


@routes.get("/prompt_library/history/{prompt_id}")
async def get_history(request):
    pid = _safe_id(request.match_info.get("prompt_id", ""))
    if not pid:
        return web.json_response({"error": "invalid id"}, status=400)
    with _lock:
        item = next((i for i in _load() if i.get("id") == pid), None)
    if item is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response({
        "id": pid,
        "history": list(reversed(item.get("history") or [])),
    })


@routes.post("/prompt_library/revert")
async def revert_prompt(request):
    payload = await request.json()
    pid = _safe_id((payload.get("id") or "").strip())
    ts = payload.get("ts")
    if not pid or ts is None:
        return web.json_response({"error": "id and ts required"}, status=400)

    with _lock:
        items = _load()
        item = next((i for i in items if i.get("id") == pid), None)
        if item is None:
            return web.json_response({"error": "not found"}, status=404)
        snap = next((s for s in item.get("history") or [] if s.get("ts") == ts), None)
        if snap is None:
            return web.json_response({"error": "snapshot not found"}, status=404)

        # Save the current state so the revert itself is undoable.
        _maybe_push_history(item, snap.get("name", ""), snap.get("text", ""), snap.get("tags") or [])
        item["name"] = snap.get("name", "")
        item["text"] = snap.get("text", "")
        item["tags"] = list(snap.get("tags") or [])
        _touch(item, created=False)
        _save(items)

    _notify_change()
    return web.json_response({
        "id": pid,
        "name": item["name"],
        "text": item["text"],
        "tags": item["tags"],
    })


@routes.post("/prompt_library/delete")
async def delete_prompt(request):
    payload = await request.json()
    pid = _safe_id((payload.get("id") or "").strip())
    if not pid:
        return web.json_response({"error": "invalid id"}, status=400)
    with _lock:
        items = [i for i in _load() if i.get("id") != pid]
        _save(items)
        _delete_image_files(pid)
    _notify_change()
    return web.json_response({"ok": True})


__version__ = "0.6.0"

NODE_CLASS_MAPPINGS = {
    "PromptLibrary": PromptLibrary,
    "PromptLibrarySave": PromptLibrarySave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLibrary": "Prompt Library",
    "PromptLibrarySave": "Prompt Library — Save",
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
