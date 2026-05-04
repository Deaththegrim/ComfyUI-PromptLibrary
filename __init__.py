import json
import os
import re
import threading
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


def _save(items: list[dict]) -> None:
    tmp = STORE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    tmp.replace(STORE_PATH)


def _safe_id(value: str) -> str | None:
    return value if value and _ID_RE.match(value) else None


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
            "has_image": _image_path_for(pid) is not None,
        })
    return web.json_response({"prompts": out})


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
    clear_image = (reader.get("clear_image") or "") == "1"
    image_field = reader.get("image")

    if not name:
        return web.json_response({"error": "name required"}, status=400)
    if not pid:
        pid = uuid.uuid4().hex[:12]
    if not _safe_id(pid):
        return web.json_response({"error": "invalid id"}, status=400)

    with _lock:
        items = _load()
        existing = next((i for i in items if i.get("id") == pid), None)
        if existing is None:
            existing = {"id": pid}
            items.append(existing)
        existing["name"] = name
        existing["text"] = text

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

    return web.json_response({
        "id": pid,
        "name": name,
        "text": text,
        "has_image": _image_path_for(pid) is not None,
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
    return web.json_response({"ok": True})


NODE_CLASS_MAPPINGS = {"PromptLibrary": PromptLibrary}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptLibrary": "Prompt Library"}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
