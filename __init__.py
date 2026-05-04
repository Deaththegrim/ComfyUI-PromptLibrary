import csv
import io
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
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


# Skip the watcher when imported by tests — each test re-imports the module and
# we'd accumulate dozens of daemon threads, slowing interpreter exit.
if "unittest" not in sys.modules and not os.environ.get("PROMPT_LIBRARY_NO_WATCHER"):
    _start_watcher()


class PromptLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_id": ("STRING", {"default": "", "multiline": False}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "utils"

    @staticmethod
    def _split_ids(prompt_id: str) -> list[str]:
        return [p.strip() for p in (prompt_id or "").split(",") if p.strip()]

    @classmethod
    def IS_CHANGED(cls, prompt_id, separator=", "):
        ids = cls._split_ids(prompt_id)
        with _lock:
            items = {i.get("id"): i.get("text", "") for i in _load()}
        return separator.join(items.get(pid, "") for pid in ids)

    def load_prompt(self, prompt_id: str, separator: str = ", "):
        ids = self._split_ids(prompt_id)
        with _lock:
            items = {i.get("id"): i.get("text", "") for i in _load()}
        parts = []
        missing = []
        for pid in ids:
            if pid in items:
                parts.append(items[pid])
            else:
                missing.append(pid)
        if missing:
            print(f"[PromptLibrary] no prompt with id(s)={missing!r}; skipped")
        return (separator.join(parts),)


_NAMED_REF_RE = re.compile(r"__([A-Za-z0-9_:.\-]+)__")
_CHOICE_RE = re.compile(r"\{([^{}]+)\}")
_WILDCARD_MAX_DEPTH = 8


def _resolve_named_ref(ref: str, items: list[dict], rng: random.Random) -> str | None:
    """Match a __name__ to an entry. Order: exact id, then exact tag (random pick)."""
    by_id = next((i for i in items if i.get("id") == ref), None)
    if by_id is not None:
        return by_id.get("text", "")
    matching = [i for i in items if ref in (i.get("tags") or [])]
    if matching:
        return rng.choice(matching).get("text", "")
    return None


def _expand_wildcards(text: str, items: list[dict], rng: random.Random,
                      *, expand_choices: bool = True, expand_named: bool = True) -> str:
    """Expand {a|b|c} alternatives and __id_or_tag__ refs against the library.

    Toggles let callers disable each mechanism independently. Cycle-safe: a shared
    placeholder dict across recursion levels; on hitting _WILDCARD_MAX_DEPTH all
    remaining refs are stashed as literals to bottom out. Unknown refs (no matching
    id/tag) are also left as literals.
    """
    state = {"placeholders": {}, "counter": [0]}

    def stash(m: re.Match) -> str:
        idx = state["counter"][0]
        state["counter"][0] += 1
        key = f"\x00U{idx}\x00"
        state["placeholders"][key] = m.group(0)
        return key

    def stash_unknowns(s: str) -> str:
        if not expand_named:
            return _NAMED_REF_RE.sub(stash, s)
        return _NAMED_REF_RE.sub(
            lambda m: stash(m) if _resolve_named_ref(m.group(1), items, rng) is None else m.group(0),
            s,
        )

    def expand(s: str, depth: int) -> str:
        if depth >= _WILDCARD_MAX_DEPTH:
            return _NAMED_REF_RE.sub(stash, s) if expand_named else s
        s = stash_unknowns(s)
        while True:
            if expand_named:
                named = _NAMED_REF_RE.search(s)
                if named:
                    resolved = _resolve_named_ref(named.group(1), items, rng) or ""
                    expanded = expand(resolved, depth + 1)
                    s = s[:named.start()] + stash_unknowns(expanded) + s[named.end():]
                    continue
            if expand_choices:
                choice = _CHOICE_RE.search(s)
                if choice:
                    content = choice.group(1)
                    picked = rng.choice([c.strip() for c in content.split("|")]) if "|" in content else content
                    s = s[:choice.start()] + picked + s[choice.end():]
                    continue
            break
        return s

    out = expand(text, 0)
    for key, original in state["placeholders"].items():
        out = out.replace(key, original)
    return out


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


_INT_MAX = 0xffffffffffffffff


class PromptLibraryRandom:
    """Pick a random library entry whose tags match a filter (AND across listed tags).

    Built for overnight loops: chain RandomByTag(character) + RandomByTag(background)
    + RandomByTag(action) into your sampler with control_after_generate=randomize so
    every queue draws a fresh combination from the library.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_filter": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX}),
            },
            "optional": {
                "expand_wildcards": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "id")
    FUNCTION = "pick"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, tag_filter, seed, expand_wildcards=True):
        return f"{seed}|{tag_filter}|{expand_wildcards}"

    def pick(self, tag_filter, seed, expand_wildcards=True):
        wanted = _parse_tags(tag_filter)
        with _lock:
            items = _load()
        if wanted:
            matches = [i for i in items if all(t in (i.get("tags") or []) for t in wanted)]
        else:
            matches = list(items)
        if not matches:
            print(f"[PromptLibrary] no entries match tag_filter={tag_filter!r}")
            return ("", "")
        rng = random.Random(seed)
        chosen = rng.choice(matches)
        text = chosen.get("text", "")
        if expand_wildcards:
            text = _expand_wildcards(text, items, rng)
        return (text, chosen.get("id", ""))


class PromptLibraryWildcard:
    """Expand {a|b|c} alternatives and __name__ library refs in a string.

    Useful when you want to author a template directly in the workflow rather
    than store it as a library entry. The two expansion mechanisms can be
    toggled independently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX}),
            },
            "optional": {
                "expand_choices": ("BOOLEAN", {"default": True}),
                "expand_named_refs": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "expand"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, text, seed, expand_choices=True, expand_named_refs=True):
        return f"{seed}|{expand_choices}|{expand_named_refs}|{text}"

    def expand(self, text, seed, expand_choices=True, expand_named_refs=True):
        with _lock:
            items = _load() if expand_named_refs else []
        rng = random.Random(seed)
        out = _expand_wildcards(text, items, rng,
                                expand_choices=expand_choices,
                                expand_named=expand_named_refs)
        return (out,)


routes = PromptServer.instance.routes


@routes.get("/prompt_library/list")
async def list_prompts(_request):
    with _lock:
        items = _load()
    out = []
    for idx, item in enumerate(items):
        pid = item.get("id", "")
        out.append({
            "id": pid,
            "name": item.get("name", ""),
            "text": item.get("text", ""),
            "tags": item.get("tags", []),
            "created_at": item.get("created_at", 0),
            "updated_at": item.get("updated_at", 0),
            "order": item.get("order", idx),
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


_MAX_IMPORT_ZIP_BYTES = 100 * 1024 * 1024


def _build_export_zip(items: list[dict], version: str) -> bytes:
    """Bundle prompts + thumbnail images into a zip. Returns the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "format": "grimm-ribbity-prompt-library",
            "format_version": 1,
            "exported_with": version,
            "exported_at": _now(),
            "prompts": items,
        }
        zf.writestr("prompts.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for item in items:
            pid = item.get("id")
            if not pid:
                continue
            img_path = _image_path_for(pid)
            if img_path:
                zf.write(img_path, arcname=f"images/{img_path.name}")
    return buf.getvalue()


@routes.post("/prompt_library/export")
async def export_zip(request):
    payload = await request.json() if request.body_exists else {}
    requested = payload.get("ids") or []
    with _lock:
        items = _load()
    if requested:
        wanted = set(requested)
        items = [i for i in items if i.get("id") in wanted]
    data = _build_export_zip(items, __version__)
    fname = f"grimmribbity-export-{int(_now())}.zip"
    return web.Response(body=data, headers={
        "Content-Type": "application/zip",
        "Content-Disposition": f'attachment; filename="{fname}"',
        "X-GrimmRibbity-Count": str(len(items)),
    })


def _import_zip(zip_bytes: bytes) -> tuple[int, int, list[str]]:
    """Import a GrimmRibbity export zip. Returns (added, updated, errors)."""
    errors: list[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return 0, 0, ["not a valid zip file"]

    with zf:
        try:
            manifest_raw = zf.read("prompts.json").decode("utf-8")
            manifest = json.loads(manifest_raw)
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            return 0, 0, [f"missing or invalid prompts.json in zip: {e}"]

        new_prompts = manifest.get("prompts") if isinstance(manifest, dict) else None
        if not isinstance(new_prompts, list):
            return 0, 0, ["prompts.json has no 'prompts' list"]

        added = updated = 0
        names_in_zip = set(zf.namelist())

        with _lock:
            items = _load()
            index = {i.get("id"): i for i in items}

            for raw in new_prompts:
                if not isinstance(raw, dict):
                    errors.append("skipping non-dict entry")
                    continue
                pid = (raw.get("id") or "").strip()
                name = (raw.get("name") or "").strip()
                if not _safe_id(pid):
                    errors.append(f"skipping entry with invalid id {pid!r}")
                    continue
                if not name:
                    errors.append(f"skipping entry id={pid!r} with empty name")
                    continue

                text_val = raw.get("text", "") or ""
                tags = _parse_tags(raw.get("tags") or [])
                history = raw.get("history") if isinstance(raw.get("history"), list) else None

                existing = index.get(pid)
                created = existing is None
                if created:
                    existing = {"id": pid}
                    items.append(existing)
                    index[pid] = existing
                    added += 1
                else:
                    _maybe_push_history(existing, name, text_val, tags)
                    updated += 1

                existing["name"] = name
                existing["text"] = text_val
                existing["tags"] = tags
                if history is not None and not created:
                    # Merge histories (incoming first, then existing); cap.
                    merged = list(history) + list(existing.get("history") or [])
                    existing["history"] = merged[-_HISTORY_CAP:]
                elif history is not None:
                    existing["history"] = list(history)[-_HISTORY_CAP:]
                _touch(existing, created=created)

                # Restore thumbnail if present in zip.
                for ext in _ALLOWED_IMAGE_EXT:
                    arc = f"images/{pid}{ext}"
                    if arc in names_in_zip:
                        try:
                            img_data = zf.read(arc)
                        except KeyError:
                            break
                        if len(img_data) > _MAX_IMAGE_BYTES:
                            errors.append(f"image for {pid!r} too large, skipped")
                            break
                        _delete_image_files(pid)
                        (IMAGES_DIR / f"{pid}{ext}").write_bytes(img_data)
                        break

            _save(items)

    return added, updated, errors


@routes.post("/prompt_library/import_zip")
async def import_zip_route(request):
    reader = await request.post()
    field = reader.get("file")
    if field is None or not hasattr(field, "file"):
        return web.json_response({"error": "no zip file provided"}, status=400)
    body = field.file.read()
    if len(body) > _MAX_IMPORT_ZIP_BYTES:
        return web.json_response({"error": "zip too large"}, status=400)
    added, updated, errors = _import_zip(body)
    _notify_change()
    return web.json_response({"added": added, "updated": updated, "errors": errors})


# Files we never want to ingest from a Prompt Builder zip:
#   - the "_Master_Filtered*" union files (they duplicate the per-category files)
#   - the user-managed Custom / Deleted lists (empty by design)
#   - anything under "_Original Files (Backup)/" (exact dupes)
_TAG_PACK_SKIP_NAMES = {"Tags-Custom.json", "Tags-Deleted.json"}
_TAG_PACK_SKIP_PREFIXES = ("Tags-_Master",)
_TAG_PACK_SKIP_PATH_PARTS = {"_Original Files (Backup)"}


def _tag_pack_extra_tags(filename: str) -> list[str]:
    """Filename-driven hints: anime files get 'anime', men files 'men', neg 'negative'."""
    extras: list[str] = []
    lower = filename.lower()
    if "anime" in lower:
        extras.append("anime")
    if lower.startswith("tags-men"):
        extras.append("men")
    if "negadvancedstyle" in lower:
        extras.append("negative")
    return extras


def _import_tag_pack_zip(zip_bytes: bytes) -> tuple[int, int, list[str]]:
    """Import a Prompt Builder zip (Tags-*.json files). Returns (added, updated, errors)."""
    errors: list[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return 0, 0, ["not a valid zip file"]

    added = updated = 0
    with zf:
        members = [n for n in zf.namelist() if n.endswith(".json")]
        with _lock:
            items = _load()
            index = {i.get("id"): i for i in items}

            for member in members:
                parts = member.split("/")
                base = parts[-1]
                if not base.startswith("Tags-"):
                    continue
                if base in _TAG_PACK_SKIP_NAMES:
                    continue
                if any(base.startswith(p) for p in _TAG_PACK_SKIP_PREFIXES):
                    continue
                if any(part in _TAG_PACK_SKIP_PATH_PARTS for part in parts[:-1]):
                    continue

                try:
                    raw = zf.read(member).decode("utf-8")
                    entries = json.loads(raw)
                except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    errors.append(f"{member}: parse failed ({e})")
                    continue
                if not isinstance(entries, list):
                    errors.append(f"{member}: not a list")
                    continue

                file_stem = base[len("Tags-"):-len(".json")]
                file_tag = _slugify(file_stem) or "prompt-builder"
                extras = _tag_pack_extra_tags(base)

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = (entry.get("name") or "").strip()
                    text_val = (entry.get("prompt") or "").strip()
                    if not name or not text_val:
                        continue
                    category = (entry.get("category") or "").strip()
                    cat_tag = _slugify(category) if category else ""
                    tags = ["prompt-builder", file_tag]
                    if cat_tag and cat_tag not in tags:
                        tags.append(cat_tag)
                    for ex in extras:
                        if ex not in tags:
                            tags.append(ex)
                    tags = _parse_tags(tags)

                    pid = _unique_id(_slugify(name) or "tag", set(index.keys()))
                    item = {"id": pid, "name": name, "text": text_val, "tags": tags}
                    _touch(item, created=True)
                    items.append(item)
                    index[pid] = item
                    added += 1

            _save(items)

    return added, updated, errors


@routes.post("/prompt_library/import_tag_packs")
async def import_tag_packs_route(request):
    reader = await request.post()
    field = reader.get("file")
    if field is None or not hasattr(field, "file"):
        return web.json_response({"error": "no zip file provided"}, status=400)
    body = field.file.read()
    if len(body) > _MAX_IMPORT_ZIP_BYTES:
        return web.json_response({"error": "zip too large"}, status=400)
    added, updated, errors = _import_tag_pack_zip(body)
    _notify_change()
    return web.json_response({"added": added, "updated": updated, "errors": errors})


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


@routes.post("/prompt_library/bulk_delete")
async def bulk_delete(request):
    payload = await request.json()
    ids_raw = payload.get("ids") or []
    valid = {i for i in (_safe_id(str(x).strip()) for x in ids_raw) if i}
    if not valid:
        return web.json_response({"error": "no valid ids"}, status=400)
    with _lock:
        items = [i for i in _load() if i.get("id") not in valid]
        _save(items)
        for pid in valid:
            _delete_image_files(pid)
    _notify_change()
    return web.json_response({"deleted": len(valid)})


@routes.post("/prompt_library/duplicate")
async def duplicate_prompt(request):
    payload = await request.json()
    pid = _safe_id((payload.get("id") or "").strip())
    if not pid:
        return web.json_response({"error": "invalid id"}, status=400)
    with _lock:
        items = _load()
        src = next((i for i in items if i.get("id") == pid), None)
        if src is None:
            return web.json_response({"error": "not found"}, status=404)
        new_name = (payload.get("name") or f"{src.get('name', '')} (copy)").strip() or "(copy)"
        new_id = _unique_id(_slugify(new_name), {i.get("id") for i in items})
        clone = {
            "id": new_id,
            "name": new_name,
            "text": src.get("text", ""),
            "tags": list(src.get("tags") or []),
        }
        _touch(clone, created=True)
        items.append(clone)
        # Copy thumbnail if present.
        src_img = _image_path_for(pid)
        if src_img is not None:
            (IMAGES_DIR / f"{new_id}{src_img.suffix}").write_bytes(src_img.read_bytes())
        _save(items)
    _notify_change()
    return web.json_response({"id": new_id, "name": new_name})


@routes.post("/prompt_library/reorder")
async def reorder_prompts(request):
    payload = await request.json()
    order_ids = payload.get("ids") or []
    if not isinstance(order_ids, list):
        return web.json_response({"error": "ids must be a list"}, status=400)
    valid = [_safe_id(str(i).strip()) for i in order_ids]
    if any(v is None for v in valid):
        return web.json_response({"error": "invalid id in list"}, status=400)
    with _lock:
        items = _load()
        rank = {pid: idx for idx, pid in enumerate(valid)}
        # Tag every item with its order; unranked items keep going at the end
        # in their existing relative order.
        max_seen = len(valid)
        for item in items:
            pid = item.get("id")
            if pid in rank:
                item["order"] = rank[pid]
            else:
                item["order"] = max_seen
                max_seen += 1
        items.sort(key=lambda i: i.get("order", 0))
        _save(items)
    _notify_change()
    return web.json_response({"ok": True, "count": len(valid)})


__version__ = "0.10.2"


def _autobackup_on_version_change() -> None:
    """Snapshot data/ to a sibling backup folder whenever __version__ changes.

    Cheap insurance against a botched upgrade. First run (no recorded version)
    skips the backup. Tests skip via the unittest gate.
    """
    marker = DATA_DIR / ".last_version"
    try:
        last = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    except OSError:
        last = ""
    if last == __version__:
        return
    has_data = STORE_PATH.exists() or any(p for p in IMAGES_DIR.iterdir() if p.name != ".gitkeep")
    if last and has_data:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = ROOT / f"data-backup-{last}-{ts}"
        try:
            shutil.copytree(DATA_DIR, backup_dir,
                            ignore=shutil.ignore_patterns("*.tmp", ".gitkeep", ".last_version"))
            print(f"[PromptLibrary] backed up data/ to {backup_dir.name} (version {last} -> {__version__})")
        except OSError as e:
            print(f"[PromptLibrary] auto-backup failed: {e}")
    try:
        marker.write_text(__version__, encoding="utf-8")
    except OSError as e:
        print(f"[PromptLibrary] could not write version marker: {e}")


if "unittest" not in sys.modules and not os.environ.get("PROMPT_LIBRARY_NO_WATCHER"):
    _autobackup_on_version_change()

NODE_CLASS_MAPPINGS = {
    "PromptLibrary": PromptLibrary,
    "PromptLibrarySave": PromptLibrarySave,
    "PromptLibraryRandom": PromptLibraryRandom,
    "PromptLibraryWildcard": PromptLibraryWildcard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLibrary": "GrimmRibbity — Library",
    "PromptLibrarySave": "GrimmRibbity — Save",
    "PromptLibraryRandom": "GrimmRibbity — Random by Tag",
    "PromptLibraryWildcard": "GrimmRibbity — Wildcard Expand",
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
