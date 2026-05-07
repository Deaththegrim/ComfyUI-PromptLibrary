import copy
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

try:
    from server import PromptServer
except ImportError:
    # Allow standalone import for offline tools (the shrink_thumbnails
    # maintenance script doesn't need ComfyUI's server module).
    PromptServer = None

try:
    import folder_paths
except ImportError:
    # Standalone import (tests, maintenance) — LoRA scanning is unavailable
    # but the rest of the package keeps working.
    folder_paths = None

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
STORE_PATH = DATA_DIR / "prompts.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Configuration — adjust these if your library outgrows the defaults.
# Wildcard-engine regexes + depth live near the engine itself further down
# (see _NAMED_REF_RE / _CHOICE_RE / _WILDCARD_MAX_DEPTH).
# =============================================================================
_lock = threading.Lock()
# Thumbnail / image storage
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MAX_IMAGE_BYTES = 16 * 1024 * 1024            # per-thumbnail upload cap
# Thumbnails are gallery icons, not source art — cap the longest edge so the
# library + zip exports stay small. The cap matches the gallery's max tile
# size (400px), so a thumbnail rendered at the largest tile setting is
# 1:1 pixel-for-pixel without upscale or wasted resolution.
_THUMBNAIL_MAX_EDGE = 400
_THUMBNAIL_JPEG_QUALITY = 80
# Export ships smaller thumbnails for transport — the on-disk copies stay
# at the storage settings above, but the export zip recompresses each one
# below so a shared library doesn't run hundreds of MB.
_EXPORT_THUMBNAIL_MAX_EDGE = 384
_EXPORT_THUMBNAIL_JPEG_QUALITY = 75
# Import / export
_MAX_IMPORT_ZIP_BYTES = 500 * 1024 * 1024      # compressed size cap on the upload
# Uncompressed-size cap on import_zip. A small zip can claim to expand
# to GB; checked against the sum of ZipInfo.file_size BEFORE any member
# is read. 2 GB swallows reasonable libraries (10k entries × ~150 KB
# thumbnails) while refusing the obvious zip-bomb shapes.
_MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
# CSV import cap. Image and zip uploads were capped, csv was not — a
# multi-GB CSV would parse into Python memory unchecked. 50 MB fits a
# library of ~200k single-line entries (the tag-pack imports are at
# this scale at the high end).
_MAX_IMPORT_CSV_BYTES = 50 * 1024 * 1024
# Per-entry history (revert disclosure in the modal)
_HISTORY_CAP = 20
# Pre-import snapshots — bulk imports save prompts.json under data/snapshots/
# before mutating, so the user can undo a destructive import. Older snapshots
# beyond _SNAPSHOT_CAP are pruned automatically.
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
_SNAPSHOT_CAP = 10
# IDs: short URL-safe slug, used as a filename component for thumbnails so
# anything outside this charset is rejected to prevent path traversal.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _load() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return _recover_corrupt_store(e)
    except OSError as e:
        print(f"[PromptLibrary] failed to read {STORE_PATH}: {e}; treating as empty")
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        items = data["prompts"]
        print(f"[PromptLibrary] {STORE_PATH.name} was an export manifest (format={data.get('format')!r}, {len(items)} prompts); rewriting as storage list")
        _snapshot_prompts("pre_heal_manifest")
        try:
            _save(items)
        except OSError as e:
            print(f"[PromptLibrary] could not rewrite {STORE_PATH}: {e}")
        return items
    return []


def _recover_corrupt_store(err: Exception) -> list[dict]:
    """Quarantine an unparseable prompts.json and try the newest snapshot.
    Without this, corruption is indistinguishable from an empty library."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    quarantined = STORE_PATH.with_name(f"{STORE_PATH.name}.broken-{ts}")
    try:
        STORE_PATH.rename(quarantined)
        print(f"[PromptLibrary] {STORE_PATH.name} unparseable ({err}); moved to {quarantined.name}")
    except OSError as oe:
        print(f"[PromptLibrary] {STORE_PATH.name} unparseable; quarantine failed: {oe}; treating as empty")
        return []
    if not SNAPSHOT_DIR.exists():
        return []
    snaps = sorted(SNAPSHOT_DIR.glob("*.json"))
    if not snaps:
        return []
    newest = snaps[-1]
    try:
        shutil.copy2(newest, STORE_PATH)
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as re:
        print(f"[PromptLibrary] snapshot {newest.name} unusable: {re}; treating as empty")
        return []
    if isinstance(data, list):
        print(f"[PromptLibrary] restored prompts.json from snapshot {newest.name} ({len(data)} entries)")
        return data
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        items = data["prompts"]
        print(f"[PromptLibrary] restored prompts.json from snapshot {newest.name} (manifest, {len(items)} entries)")
        _save(items)
        return items
    return []


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


def _snapshot_prompts(label: str) -> str:
    """Save a copy of the current prompts.json under data/snapshots/ so the
    user can roll back a destructive bulk operation. Returns the snapshot
    filename (basename only). Old snapshots beyond _SNAPSHOT_CAP are pruned."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        return ""
    safe_label = "".join(c if c.isalnum() else "_" for c in (label or "import"))[:32]
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{safe_label}.json"
    out = SNAPSHOT_DIR / name
    try:
        shutil.copy2(STORE_PATH, out)
    except OSError as e:
        print(f"[PromptLibrary] snapshot failed: {e}")
        return ""
    # Prune older snapshots beyond the cap
    try:
        snaps = sorted(SNAPSHOT_DIR.glob("*.json"))
        for old in snaps[:-_SNAPSHOT_CAP]:
            try:
                old.unlink()
            except OSError:
                pass
    except OSError:
        pass
    return name


def _list_snapshots() -> list[dict]:
    """Return the available snapshots, newest first, with parsed metadata."""
    if not SNAPSHOT_DIR.exists():
        return []
    out = []
    for path in sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True):
        try:
            stat = path.stat()
        except OSError:
            continue
        # Filename: <YYYYMMDD-HHMMSS>-<label>.json
        stem = path.stem
        ts_part, _, label = stem.partition("-")
        if len(ts_part) == 8 and "-" in stem:
            # New format: YYYYMMDD-HHMMSS-label
            date_part, _, rest = stem.partition("-")
            time_part, _, label = rest.partition("-")
            label = label or "import"
        else:
            label = "snapshot"
        out.append({
            "name": path.name,
            "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
            "label": label,
        })
    return out


def _restore_snapshot(name: str) -> dict:
    """Restore prompts.json from a snapshot. Returns a dict describing what
    happened. Refuses paths with separators (no escapes outside the
    snapshot dir) AND verifies the resolved path is actually rooted inside
    SNAPSHOT_DIR — handles symlink edge cases the substring check would
    otherwise miss."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return {"ok": False, "error": f"invalid snapshot name {name!r}"}
    src = SNAPSHOT_DIR / name
    # Defence in depth: even with the substring check above, a symlink in
    # SNAPSHOT_DIR pointing outside (placed by a separate process) could
    # let _restore_snapshot copy from any file the Comfy process can read.
    # resolve() chases symlinks; is_relative_to() catches the escape.
    try:
        resolved = src.resolve(strict=False)
        snapshot_root = SNAPSHOT_DIR.resolve(strict=False)
        if not resolved.is_relative_to(snapshot_root):
            return {"ok": False,
                    "error": f"snapshot {name!r} resolves outside the snapshots directory"}
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"could not resolve snapshot path: {e}"}
    if not src.is_file():
        return {"ok": False, "error": f"snapshot {name!r} not found"}
    # Take a "current state" snapshot too so undo is itself undoable.
    pre_undo = _snapshot_prompts("pre_undo")
    with _lock:
        try:
            shutil.copy2(src, STORE_PATH)
        except OSError as e:
            return {"ok": False, "error": f"copy failed: {e}"}
        # Force-update mtime tracker so the watcher fires the refresh event.
        global _last_known_mtime
        try:
            _last_known_mtime = STORE_PATH.stat().st_mtime
        except OSError:
            pass
        items = _load()
    return {
        "ok": True,
        "restored_from": name,
        "pre_undo_snapshot": pre_undo,
        "entries": len(items),
    }


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


_LORAS_PER_ENTRY_CAP = 10


def _parse_loras(value) -> list[dict]:
    """Normalise the modal's LoRA payload into the on-disk shape.

    Accepts:
      - a JSON-encoded string (what the modal sends inside multipart form data)
      - a list of dicts (what zip-import / direct API callers send)
    Returns a list of {name, strength_model, strength_clip, triggers, enabled}
    capped at _LORAS_PER_ENTRY_CAP. Rows missing a `name` are dropped silently
    so the modal can keep an empty placeholder row without writing junk to disk.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
    else:
        data = value
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for raw in data[:_LORAS_PER_ENTRY_CAP]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        try:
            sm = float(raw.get("strength_model", raw.get("strength", 1.0)) or 0.0)
        except (TypeError, ValueError):
            sm = 1.0
        # If only one strength is supplied, mirror it onto CLIP — matches
        # the comfy LoraLoader UX where the two sliders default in lockstep.
        try:
            sc = float(raw.get("strength_clip", raw.get("strength", sm)) or 0.0)
        except (TypeError, ValueError):
            sc = sm
        sm = max(-2.0, min(2.0, sm))
        sc = max(-2.0, min(2.0, sc))
        triggers = str(raw.get("triggers") or "").strip()
        enabled = bool(raw.get("enabled", True))
        out.append({
            "name": name,
            "strength_model": sm,
            "strength_clip": sc,
            "triggers": triggers,
            "enabled": enabled,
        })
    return out


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


def _save_thumbnail_pil(prompt_id: str, pil) -> str | None:
    """Downscale a PIL image and write it as a thumbnail. Returns the saved
    extension ('.png' or '.jpg') on success, None on failure.

    PNG when the image has alpha (transparency would be lost in JPEG), else
    JPEG at q=85 — gives a ~5-15× smaller file than the original PNG.
    """
    try:
        from PIL import Image  # noqa: F401
    except ImportError as e:
        print(f"[PromptLibrary] PIL unavailable, can't save thumbnail: {e}")
        return None
    try:
        pil.thumbnail((_THUMBNAIL_MAX_EDGE, _THUMBNAIL_MAX_EDGE))
        has_alpha = pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info)
        _delete_image_files(prompt_id)
        if has_alpha:
            if pil.mode != "RGBA":
                pil = pil.convert("RGBA")
            pil.save(IMAGES_DIR / f"{prompt_id}.png", format="PNG", optimize=True)
            return ".png"
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil.save(IMAGES_DIR / f"{prompt_id}.jpg", format="JPEG",
                 quality=_THUMBNAIL_JPEG_QUALITY, optimize=True, progressive=True)
        return ".jpg"
    except Exception as e:
        print(f"[PromptLibrary] failed to save thumbnail for {prompt_id!r}: {e}")
        return None


def _save_thumbnail_bytes(prompt_id: str, data: bytes) -> str | None:
    """Open arbitrary image bytes and save as a downscaled thumbnail."""
    try:
        from PIL import Image
    except ImportError as e:
        print(f"[PromptLibrary] PIL unavailable, can't save thumbnail: {e}")
        return None
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception as e:
        print(f"[PromptLibrary] failed to decode image for {prompt_id!r}: {e}")
        return None
    return _save_thumbnail_pil(prompt_id, pil)


def _save_image_tensor(prompt_id: str, image) -> bool:
    """Save the first frame of a ComfyUI IMAGE batch as a downscaled thumbnail."""
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
        return _save_thumbnail_pil(prompt_id, pil) is not None
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


def _push_history(item: dict) -> None:
    """Snapshot the current name/text/tags/loras onto the entry's history list."""
    history = list(item.get("history") or [])
    snap: dict = {
        "ts": _now(),
        "name": item.get("name", ""),
        "text": item.get("text", ""),
        "tags": list(item.get("tags") or []),
    }
    if item.get("loras"):
        snap["loras"] = [dict(l) for l in item["loras"]]
    history.append(snap)
    item["history"] = history[-_HISTORY_CAP:]


def _maybe_push_history(item: dict, new_name: str, new_text: str, new_tags: list,
                          new_negative: str = "", new_loras: list | None = None) -> None:
    """Push history only if any user-visible field actually changes (image excluded)."""
    if (item.get("name", "") == new_name
        and item.get("text", "") == new_text
        and item.get("negative", "") == new_negative
        and list(item.get("tags") or []) == list(new_tags or [])
        and list(item.get("loras") or []) == list(new_loras or [])):
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
        # Debounce window: when external edits land in a tight burst (e.g. a
        # script writing several entries in <2 s), we'd otherwise emit a
        # websocket refresh per change. Wait for the mtime to settle for
        # one full tick before notifying — coalesces the burst into one
        # gallery refresh on every connected client.
        pending_mtime = 0.0
        while True:
            time.sleep(2)
            # Two-layer guard: the inner try/except OSError handles the
            # narrow stat-failure case; the outer try/except Exception is
            # the survival net so a future bug in _notify_change can't
            # silently kill the thread and break gallery auto-refresh for
            # the rest of the Comfy session.
            try:
                try:
                    mtime = STORE_PATH.stat().st_mtime if STORE_PATH.exists() else 0.0
                except OSError:
                    continue
                if mtime == _last_known_mtime:
                    continue
                if pending_mtime != mtime:
                    # New change observed — wait one more tick to coalesce.
                    pending_mtime = mtime
                    continue
                # Same mtime as last tick — file has settled, fire.
                _last_known_mtime = mtime
                _notify_change()
            except Exception as e:
                print(f"[PromptLibrary] watcher iteration failed: {e!r}; continuing")

    t = threading.Thread(target=loop, daemon=True, name="prompt-library-watcher")
    t.start()


# Skip the watcher when imported by tests — each test re-imports the module and
# we'd accumulate dozens of daemon threads, slowing interpreter exit.
if "unittest" not in sys.modules and not os.environ.get("PROMPT_LIBRARY_NO_WATCHER"):
    _start_watcher()


class PromptLibrary:
    DESCRIPTION = (
        "Visual prompt picker. Browse the gallery, click tiles to select one "
        "or many entries, and the joined prompt text is emitted on the output. "
        "Multi-select is persistent across filter/search changes. "
        "If MODEL + CLIP are wired in, the LoRA stacks attached to every "
        "selected entry (set in the Edit Prompt modal) are applied in "
        "selection order — single node replaces a Library + Power Lora "
        "Loader chain."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_id": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated entry IDs. Driven by the gallery widget — "
                               "you don't normally type here, but you can paste IDs to pre-select."}),
                "separator": ("STRING", {"default": ", ", "multiline": False,
                    "tooltip": "Glue between joined entries when multiple tiles are selected."}),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Optional. Wire to enable LoRA application. The selected "
                               "entries' LoRA stacks are applied to this MODEL in pick-order; "
                               "the patched MODEL is emitted on the matching output. Leave "
                               "unwired for STRING-only behaviour (the default)."}),
                "clip": ("CLIP", {
                    "tooltip": "Optional. Required alongside MODEL for LoRA application. "
                               "Patched CLIP is emitted on the matching output."}),
                "strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier applied to every LoRA's strength_model + strength_clip "
                               "for this run. 0.0 disables all LoRAs without unwiring MODEL/CLIP."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "MODEL", "CLIP")
    RETURN_NAMES = ("prompt", "negative", "model", "clip")
    OUTPUT_TOOLTIPS = (
        "The selected prompt(s) joined by the separator.",
        "Joined negative-prompt text from the same entries (empty for entries "
        "that don't store one). Wire into the negative side of your sampler.",
        "MODEL with every selected entry's LoRA stack applied. None if MODEL "
        "input was unwired.",
        "CLIP with every selected entry's LoRA stack applied. None if CLIP "
        "input was unwired.",
    )
    FUNCTION = "load_prompt"
    CATEGORY = "GrimmRibbity/Library"

    @staticmethod
    def _split_ids(prompt_id: str) -> list[str]:
        return [p.strip() for p in (prompt_id or "").split(",") if p.strip()]

    @classmethod
    def IS_CHANGED(cls, prompt_id, separator=", ", model=None, clip=None,
                    strength_scale=1.0):
        ids = cls._split_ids(prompt_id)
        with _lock:
            items = {i.get("id"): i for i in _load()}
        # Include LoRA signatures in the hash only when MODEL/CLIP would
        # actually use them — otherwise editing an entry's LoRA stack
        # shouldn't re-trigger pure STRING-only consumers.
        lora_aware = model is not None and clip is not None
        parts: list[str] = []
        for pid in ids:
            entry = items.get(pid)
            if entry is None:
                parts.append(f"miss:{pid}")
                continue
            text = entry.get("text", "")
            neg = entry.get("negative", "")
            if lora_aware:
                loras = entry.get("loras") or []
                lora_sig = "|".join(
                    f"{l.get('name','')}:{l.get('strength_model',1.0):g}:"
                    f"{l.get('strength_clip',1.0):g}:"
                    f"{int(bool(l.get('enabled', True)))}:"
                    f"{(l.get('triggers') or '').strip()}"
                    for l in loras
                )
                parts.append(f"{text}::{neg}::{lora_sig}")
            else:
                parts.append(f"{text}::{neg}")
        return (f"{separator}::{strength_scale:g}::"
                + ("L" if lora_aware else "S") + "::"
                + "@@".join(parts))

    def load_prompt(self, prompt_id: str, separator: str = ", ",
                     model=None, clip=None, strength_scale: float = 1.0):
        ids = self._split_ids(prompt_id)
        with _lock:
            items = {i.get("id"): i for i in _load()}
        pos_parts: list[str] = []
        neg_parts: list[str] = []
        entries: list[dict] = []
        missing: list[str] = []
        for pid in ids:
            entry = items.get(pid)
            if entry is None:
                missing.append(pid)
                continue
            entries.append(entry)
            text = entry.get("text", "")
            if text:
                pos_parts.append(text)
            neg = entry.get("negative", "")
            if neg:
                neg_parts.append(neg)
        if missing:
            print(f"[PromptLibrary] no prompt with id(s)={missing!r}; skipped")

        prompt_out = separator.join(pos_parts)
        neg_out = separator.join(neg_parts)

        # LoRA application only runs when both MODEL and CLIP are wired.
        # The helper lives in style_node.py — lazy-imported here so the
        # plain STRING-only flow doesn't need torch / comfy.sd at import.
        model_out, clip_out = model, clip
        # Half-wired guard: wiring MODEL without CLIP (or vice versa) used
        # to silently skip LoRA application and emit a half-patched tuple
        # downstream — confusing because the user explicitly wired the
        # model side but got no LoRA effect. Warn once with the rule.
        if (model is None) != (clip is None):
            print("[PromptLibrary] WARNING: only one of MODEL / CLIP is wired — "
                  "LoRA application requires BOTH. Wire the matching socket "
                  "(or unwire both) to get LoRA-patched output.")
        if model is not None and clip is not None and entries:
            try:
                from .style_node import _apply_loras
                patched_model, patched_clip = model, clip
                applied = 0
                status_all: list[str] = []
                for entry in entries:
                    loras = list(entry.get("loras") or [])
                    if not loras:
                        continue
                    patched_model, patched_clip, status = _apply_loras(
                        patched_model, patched_clip, loras, strength_scale)
                    applied += sum(1 for s in status if s.startswith("+"))
                    status_all.extend(status)
                model_out, clip_out = patched_model, patched_clip
                if status_all:
                    names = [e.get("name") or e.get("id", "?") for e in entries]
                    print(f"[PromptLibrary] {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
                          f"({', '.join(names)}); {applied} LoRA(s) applied: "
                          f"{' '.join(status_all)}")
            except Exception as e:
                print(f"[PromptLibrary] LoRA application failed: {e}; "
                      f"emitting unpatched MODEL/CLIP")

        return (prompt_out, neg_out, model_out, clip_out)


class PromptLibraryMulti:
    """Three independent gallery panels in a single node, so a workflow can
    pull a Character / Style / Clothing-style split selection without wiring
    three separate Library nodes + Join Strings nodes together. Each panel
    has its own search, filter, sort, and selection state, and emits its
    own STRING output."""

    PANELS = 3
    DESCRIPTION = (
        "Three independent gallery panels in a single node. Use it to replace "
        "Character / Style / Clothing → 3× Library + 2× Join Strings spaghetti "
        "with a tidy single node that emits three STRING outputs. Each panel has "
        "its own search, tag filter, sort mode, and selection. Click a panel "
        "header to rename it inline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        req: dict = {}
        for i in range(1, cls.PANELS + 1):
            req[f"label_{i}"] = ("STRING", {"default": f"Panel {i}", "multiline": False,
                "tooltip": f"Display label for panel {i}. Click the panel header in the "
                           "node UI to rename inline."})
            req[f"prompt_id_{i}"] = ("STRING", {"default": "", "multiline": False,
                "tooltip": f"Comma-separated entry IDs for panel {i}. Driven by the gallery; "
                           "saved with the workflow."})
            req[f"separator_{i}"] = ("STRING", {"default": ", ", "multiline": False,
                "tooltip": f"Glue between joined entries for panel {i}."})
        return {"required": req}

    RETURN_TYPES = tuple(["STRING"] * PANELS)
    RETURN_NAMES = tuple(f"prompt_{i}" for i in range(1, PANELS + 1))
    OUTPUT_TOOLTIPS = tuple(f"Panel {i} selection joined by separator_{i}."
                             for i in range(1, PANELS + 1))
    FUNCTION = "load_prompts"
    CATEGORY = "GrimmRibbity/Library"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        with _lock:
            items = {i.get("id"): i.get("text", "") for i in _load()}
        sigs = []
        for n in range(1, cls.PANELS + 1):
            ids = [p.strip() for p in (kwargs.get(f"prompt_id_{n}") or "").split(",") if p.strip()]
            sep = kwargs.get(f"separator_{n}", ", ")
            sigs.append(sep.join(items.get(pid, "") for pid in ids))
        return "".join(sigs)

    def load_prompts(self, **kwargs):
        with _lock:
            items = {i.get("id"): i.get("text", "") for i in _load()}
        out = []
        for n in range(1, self.PANELS + 1):
            ids = [p.strip() for p in (kwargs.get(f"prompt_id_{n}") or "").split(",") if p.strip()]
            sep = kwargs.get(f"separator_{n}", ", ")
            missing = [pid for pid in ids if pid not in items]
            if missing:
                print(f"[PromptLibraryMulti] panel {n}: no prompt with id(s)={missing!r}; skipped")
            out.append(sep.join(items[pid] for pid in ids if pid in items))
        return tuple(out)


_NAMED_REF_RE = re.compile(r"__([A-Za-z0-9_:.\-]+)__")
_CHOICE_RE = re.compile(r"\{([^{}]+)\}")
# Weighted choice prefix: "<float>::" — e.g. {0.25::a|0.75::b}. Bare alternatives
# get an implicit weight of 1.0, so {a|b|c} stays uniform. Negative or non-numeric
# prefixes fall back to weight 1.0.
_WEIGHT_PREFIX_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*::\s*(.*)$", re.DOTALL)
_WILDCARD_MAX_DEPTH = 8


def _parse_weighted_choices(content: str) -> tuple[list[str], list[float]]:
    """Split a {…|…|…} content string into (texts, weights). Each alt may be
    prefixed with `<float>::` to set its weight; missing prefix defaults to 1.0.
    Negative or unparseable prefixes are clamped to 0.0 (the literal is kept,
    just deweighted to nothing). All-zero weights fall back to uniform."""
    texts: list[str] = []
    weights: list[float] = []
    for raw in content.split("|"):
        m = _WEIGHT_PREFIX_RE.match(raw)
        if m:
            try:
                w = float(m.group(1))
            except ValueError:
                w = 1.0
            if w < 0:
                w = 0.0
            texts.append(m.group(2).strip())
            weights.append(w)
        else:
            texts.append(raw.strip())
            weights.append(1.0)
    if sum(weights) <= 0:
        weights = [1.0] * len(texts)
    return texts, weights


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
                    if "|" in content:
                        texts, weights = _parse_weighted_choices(content)
                        picked = rng.choices(texts, weights=weights, k=1)[0]
                    else:
                        picked = content
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

    DESCRIPTION = (
        "Save a prompt to the GrimmRibbity library when the workflow runs. "
        "Wire any STRING source into 'text' and an IMAGE source into 'thumbnail'. "
        "On Queue, the entry is appended (or updated, if you provide prompt_id "
        "or set overwrite_by_name). The open gallery widgets refresh "
        "automatically via websocket. Errors on empty 'name' or 'text'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Display name shown on tiles and used for slug-based "
                               "ID auto-generation. Required."}),
                "text": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Prompt text to store. Required — empty values raise an error so "
                               "an unwired input doesn't silently save a blank entry."}),
            },
            "optional": {
                "thumbnail": ("IMAGE", {"tooltip": "Optional preview image. The first frame is "
                                                      "downsized to <=512px and stored as a thumbnail."}),
                "negative": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Optional negative-prompt text stored alongside this entry. "
                               "The Library loader emits it on its 'negative' output port."}),
                "tags": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated tags. Use 'category:value' (e.g. model:anima, "
                               "style:cyberpunk) to enable category-grouped filter chips."}),
                "prompt_id": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Override the auto-derived ID. Must match [A-Za-z0-9_-]{1,64}. "
                               "Leave blank to slug from 'name'."}),
                "overwrite_by_name": ("BOOLEAN", {"default": False,
                    "tooltip": "When prompt_id is unset and an entry with this name already exists, "
                               "update that entry instead of inserting a new one."}),
                "loras_json": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "[] — leave blank to leave loras untouched",
                    "tooltip": "Optional JSON-encoded LoRA stack to attach to the saved entry. "
                               "Same shape as the modal's loras list: [{\"name\": \"path/to.safetensors\", "
                               "\"strength_model\": 1.0, \"strength_clip\": 1.0, \"triggers\": \"...\", "
                               "\"enabled\": true}]. Leave blank to keep whatever loras the entry "
                               "already has (or none, for a brand-new entry). Capped at 10 rows."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "id")
    OUTPUT_TOOLTIPS = (
        "The saved prompt text (passthrough of the input).",
        "The entry's ID — wire into the loader's prompt_id or use for re-runs.",
    )
    FUNCTION = "save"
    CATEGORY = "GrimmRibbity/Library"
    OUTPUT_NODE = True

    def save(self, name, text, thumbnail=None, negative="", tags="", prompt_id="",
              overwrite_by_name=False, loras_json=""):
        name = (name or "").strip()
        if not name:
            raise ValueError("PromptLibrarySave: name is required")
        if not (text or "").strip():
            raise ValueError("PromptLibrarySave: prompt text is empty — connect a STRING source or type something into the text field")
        prompt_id = (prompt_id or "").strip()
        if prompt_id and not _safe_id(prompt_id):
            raise ValueError(f"PromptLibrarySave: invalid prompt_id {prompt_id!r}")
        parsed_tags = _parse_tags(tags)
        # Parse loras_json with the same normaliser the upsert route uses,
        # so the workflow-side JSON shape matches what the modal stores.
        # Empty string ⇒ untouched (None signals "don't write the field");
        # an explicit "[]" ⇒ clear the entry's loras list.
        loras_provided = bool((loras_json or "").strip())
        loras = _parse_loras(loras_json) if loras_provided else None

        with _lock:
            items = _load()
            existing = None
            match_path = "new"
            if prompt_id:
                existing = next((i for i in items if i.get("id") == prompt_id), None)
                match_path = "prompt_id" if existing else "new (id miss)"
            elif overwrite_by_name:
                # Prefer the most-recently-updated entry when multiple share the
                # same name. The previous code returned the first match by
                # on-disk insertion order, which silently always-targets the
                # OLDEST duplicate — surprising when a user expects "the one I
                # was just editing" semantics. Picking by updated_at matches
                # the human intent and is stable when no dupes exist.
                matches = [i for i in items if i.get("name") == name]
                if matches:
                    existing = max(matches, key=lambda i: i.get("updated_at", 0))
                    match_path = (f"overwrite_by_name (1 of {len(matches)} matches"
                                   if len(matches) > 1 else "overwrite_by_name")

            created = existing is None
            if created:
                pid = prompt_id or _unique_id(_slugify(name), {i.get("id") for i in items})
                existing = {"id": pid}
                items.append(existing)
            else:
                _maybe_push_history(existing, name, text or "", parsed_tags,
                                      new_negative=negative or "",
                                      new_loras=loras if loras_provided
                                                else existing.get("loras"))

            existing["name"] = name
            existing["text"] = text or ""
            existing["tags"] = parsed_tags
            if negative or "negative" in existing:
                existing["negative"] = negative or ""
            if loras_provided:
                if loras:
                    existing["loras"] = loras
                elif "loras" in existing:
                    existing["loras"] = []
            _touch(existing, created=created)

            if thumbnail is not None:
                _save_image_tensor(existing["id"], thumbnail)

            _save(items)
            saved_id = existing["id"]

        _notify_change()
        # Log which lookup path was taken so a sticky prompt_id widget or an
        # ambiguous overwrite_by_name match can be diagnosed from the console
        # without rebuilding the workflow.
        print(f"[PromptLibrary] saved id={saved_id!r} name={name!r} "
              f"via={match_path} tags={parsed_tags}")
        return (text or "", saved_id)


_INT_MAX = 0xffffffffffffffff


class PromptLibraryThumbnailSaver:
    """Updates ONLY the thumbnail on an existing library entry — leaves name,
    text, tags, rating, and notes alone. Pair with the gallery's Queue ▶▶
    button: it drives this node's `prompt_id` widget alongside the Library
    node's, so each queued run thumbnails its own entry.

    Workflow shape:
      Library (gallery-driven) → text → CLIPTextEncode → Sampler → IMAGE
                                                                     ↓
                                                          ThumbnailSaver (prompt_id matched)

    Skips silently if `prompt_id` is empty (so the workflow doesn't crash
    when someone runs it interactively without the bulk-driver). Skips
    with a console note if the id doesn't resolve to an existing entry.
    """

    DESCRIPTION = (
        "Update an existing library entry's thumbnail. Wire IMAGE + the "
        "entry's prompt_id; on workflow run, the entry's thumbnail is "
        "replaced. Doesn't change the entry's name / text / tags. Use with "
        "the gallery's Queue ▶▶ button — it drives this node's prompt_id "
        "in lock-step with the Library loader."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to save as the thumbnail. The first "
                                                "frame of a batch is used unless frame_index "
                                                "is set."}),
                "prompt_id": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Library entry ID to update. Driven by Queue ▶▶ in lock-step "
                               "with the Library loader."}),
            },
            "optional": {
                "frame_index": ("INT", {"default": 1, "min": 1, "max": 4096,
                    "tooltip": "When the wired IMAGE is a batch, which frame to use as the "
                               "thumbnail (1-based)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("id",)
    OUTPUT_TOOLTIPS = ("The prompt_id that was processed, or empty string if skipped.",)
    FUNCTION = "save_thumb"
    CATEGORY = "GrimmRibbity/Library"
    OUTPUT_NODE = True

    def save_thumb(self, image, prompt_id, frame_index=1):
        pid = (prompt_id or "").strip()
        if not pid:
            print("[ThumbnailSaver] prompt_id empty; skipping (workflow probably "
                  "running interactively, not via Queue ▶▶)")
            return ("",)
        if not _safe_id(pid):
            print(f"[ThumbnailSaver] invalid prompt_id {pid!r}; skipping")
            return ("",)

        with _lock:
            items = _load()
            if not any(i.get("id") == pid for i in items):
                print(f"[ThumbnailSaver] no entry with id={pid!r}; skipping")
                return ("",)

        # Encode the chosen frame to PNG bytes. _save_thumbnail_bytes handles
        # the downscale + JPEG re-encode for storage.
        from PIL import Image
        import numpy as np
        idx = max(0, min(int(frame_index) - 1, image.shape[0] - 1))
        frame = image[idx]
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        arr = (frame.clip(0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        if _save_thumbnail_bytes(pid, buf.getvalue()) is None:
            print(f"[ThumbnailSaver] failed to process thumbnail for {pid!r}")
            return ("",)
        # Bump the entry's updated_at so the watcher fires + UIs refresh.
        entry_name = ""
        with _lock:
            items = _load()
            entry = next((i for i in items if i.get("id") == pid), None)
            if entry is not None:
                entry_name = entry.get("name", "")
                _touch(entry, created=False)
                _save(items)
        _notify_change()
        # Log the targeted entry on success so a sticky prompt_id widget on
        # this node (which would clobber the same entry's thumbnail every
        # run) is visible from the Comfy console — same diagnostic the
        # PromptLibrarySave node prints.
        print(f"[ThumbnailSaver] wrote thumbnail to id={pid!r} name={entry_name!r}")
        return (pid,)


class PromptLibraryRandom:
    """Pick a random library entry whose tags match a filter (AND across listed tags).

    Built for overnight loops: chain RandomByTag(character) + RandomByTag(background)
    + RandomByTag(action) into your sampler with control_after_generate=randomize so
    every queue draws a fresh combination from the library.
    """

    DESCRIPTION = (
        "Pick a random library entry whose tags match an AND filter. Set "
        "control_after_generate=randomize on the seed to draw a fresh entry "
        "every queue — useful for overnight loops where you chain "
        "RandomByTag(character) + RandomByTag(background) + RandomByTag(action)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag_filter": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated tags that EVERY matched entry must have. "
                               "Empty matches all entries."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX,
                    "tooltip": "Set control_after_generate=randomize for fresh picks per queue. "
                               "Same seed + same filter = deterministic pick."}),
            },
            "optional": {
                "expand_wildcards": ("BOOLEAN", {"default": True,
                    "tooltip": "Expand {a|b|c} alternatives and __id_or_tag__ refs in the picked text."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "id", "negative")
    OUTPUT_TOOLTIPS = (
        "The randomly picked entry's prompt text (after wildcard expansion).",
        "The picked entry's ID — useful for logging or re-running.",
        "The entry's negative-prompt text (empty if none stored).",
    )
    FUNCTION = "pick"
    CATEGORY = "GrimmRibbity/Library"

    @classmethod
    def IS_CHANGED(cls, tag_filter, seed, expand_wildcards=True):
        # Hash the prompts.json mtime into the cache key so that editing a
        # tag-matched entry mid-session invalidates the cached pick. Without
        # this, a fixed-seed Random by Tag run would keep emitting the
        # stale text from a previous render even after the user updated
        # the entry. control_after_generate=randomize already invalidates
        # via the seed bump, but interactive workflows benefit too.
        try:
            mtime = STORE_PATH.stat().st_mtime if STORE_PATH.exists() else 0.0
        except OSError:
            mtime = 0.0
        return f"{seed}|{tag_filter}|{expand_wildcards}|{mtime}"

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
            return ("", "", "")
        rng = random.Random(seed)
        chosen = rng.choice(matches)
        text = chosen.get("text", "")
        negative = chosen.get("negative", "")
        if expand_wildcards:
            text = _expand_wildcards(text, items, rng)
            if negative:
                negative = _expand_wildcards(negative, items, rng)
        return (text, chosen.get("id", ""), negative)


class PromptLibraryWildcard:
    """Expand {a|b|c} alternatives and __name__ library refs in a string.

    Useful when you want to author a template directly in the workflow rather
    than store it as a library entry. The two expansion mechanisms can be
    toggled independently.
    """

    DESCRIPTION = (
        "Expand {a|b|c} alternatives and __id_or_tag__ library refs in a string. "
        "Supports weighted choices ({0.25::a|0.75::b}). Cycle-safe (max depth 8). "
        "Use when you want to author a template directly in the workflow rather "
        "than save it as a library entry."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Template text. {a|b|c} picks one alternative; "
                               "{0.25::a|0.75::b} weighted; __knight__ pulls from a library entry "
                               "with id 'knight' or a random entry tagged 'knight'."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX,
                    "tooltip": "Drives all random picks. Same seed = same expansion."}),
            },
            "optional": {
                "expand_choices": ("BOOLEAN", {"default": True,
                    "tooltip": "Toggle {a|b|c} alternation. Off = leave braces literal."}),
                "expand_named_refs": ("BOOLEAN", {"default": True,
                    "tooltip": "Toggle __name__ library refs. Off = leave them literal."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("Expanded text. Unknown refs and unmatched braces are left as-is.",)
    FUNCTION = "expand"
    CATEGORY = "GrimmRibbity/Library"

    @classmethod
    def IS_CHANGED(cls, text, seed, expand_choices=True, expand_named_refs=True):
        # When expand_named_refs is on, the result depends on the live
        # library content (which entries match a __name__ reference),
        # so editing a referenced entry mid-session must invalidate the
        # cache. Folds the prompts.json mtime into the hash on that
        # branch only — pure {a|b|c} expansion that doesn't touch the
        # library doesn't pay the cache invalidation cost.
        lib_sig = ""
        if expand_named_refs:
            try:
                lib_sig = str(STORE_PATH.stat().st_mtime if STORE_PATH.exists() else 0.0)
            except OSError:
                lib_sig = ""
        return f"{seed}|{expand_choices}|{expand_named_refs}|{lib_sig}|{text}"

    def expand(self, text, seed, expand_choices=True, expand_named_refs=True):
        with _lock:
            items = _load() if expand_named_refs else []
        rng = random.Random(seed)
        out = _expand_wildcards(text, items, rng,
                                expand_choices=expand_choices,
                                expand_named=expand_named_refs)
        return (out,)


# =============================================================================
# Comic-strip authoring: Scene + Background anchors + per-frame assembler.
# =============================================================================


_SCENE_NONE = "(none)"
# Curated dropdown vocabularies for the Scene node. First entry of each
# list is the "skip" sentinel and gets filtered out when joining the
# prompt. The time_of_day / weather / lighting / camera_angle lists are
# aligned with Danbooru's tag groups (see
# https://danbooru.donmai.us/wiki_pages/tag_groups) so SDXL anime
# checkpoints — which are typically Danbooru-tag-trained — recognise the
# values and respond well. mood / framing are photography concepts that
# Danbooru doesn't index, kept as our own curated list.
#
# Tag-group sources used:
#   - tag_group:lighting (lighting + time of day lighting)
#   - tag_group:image_composition (view angles + framing the body)
_SCENE_TIME_OF_DAY = [_SCENE_NONE,
    # Danbooru-canonical time-of-day tags
    "dawn", "sunrise", "morning", "day", "afternoon",
    "evening", "sunset", "dusk", "twilight", "night",
    # Photography-canonical complements
    "early morning", "midday", "late afternoon",
    "golden hour", "blue hour", "midnight"]
_SCENE_WEATHER = [_SCENE_NONE,
    # Sky conditions
    "clear sky", "blue sky", "partly cloudy", "cloudy", "overcast",
    "stormy clouds",
    # Precipitation (Danbooru: rain, raining, snow, snowing)
    "light rain", "rain", "raining", "heavy rain", "drizzle",
    "thunderstorm", "lightning", "sun shower",
    "light snow", "snow", "snowing", "heavy snow", "blizzard",
    # Air conditions
    "fog", "mist", "haze",
    # Wind / dust
    "windy", "sandstorm", "dust storm",
    # Optical
    "rainbow", "aurora"]
_SCENE_LIGHTING = [_SCENE_NONE,
    # Direction (Danbooru tag_group:lighting → Directional)
    "backlighting", "sidelighting", "overlighting", "underlighting",
    "rim lighting",
    # Types (Danbooru → Types)
    "bloom", "spotlight", "light rays", "sunbeam", "moonbeam",
    "sunlight", "moonlight", "dim lighting",
    # Through gaps (Danbooru → Light Through Gaps)
    "dappled sunlight", "dappled moonlight", "crack of light",
    "window shadow",
    # Sources (Danbooru → Light Sources)
    "candlelight", "firelight", "lantern light", "lamplight",
    "torchlight", "neon lighting", "fluorescent lighting",
    "stage lighting", "headlight", "floodlights",
    # Atmosphere / volumetrics
    "volumetric lighting", "god rays", "soft lighting",
    # Shadow / contrast (Danbooru → Absence of Light)
    "harsh shadows", "soft shadows", "silhouette", "chiaroscuro",
    "shade"]
_SCENE_CAMERA_ANGLE = [_SCENE_NONE,
    # View angle (Danbooru tag_group:image_composition → View Angle)
    "from above", "from behind", "from below", "from side",
    "three-quarter view", "straight-on", "dutch angle", "upside-down",
    # Perspective / depth
    "fisheye", "panorama", "perspective", "isometric",
    # Framing the body (Danbooru → Framing the Body)
    "extreme close-up", "close-up", "portrait", "upper body",
    "cowboy shot", "full body", "wide shot", "very wide shot",
    "lower body", "profile",
    # Composition / POV
    "pov", "over-the-shoulder", "from outside",
    # Eye-line cues (photography)
    "eye level", "low angle", "high angle",
    "bird's-eye view", "worm's-eye view"]
_SCENE_MOOD = [_SCENE_NONE,
    # Mood is a photography/illustration concept Danbooru doesn't index
    # as its own tag group; keep our hand-curated list.
    "tense", "ominous", "anxious", "dramatic",
    "melancholic", "nostalgic", "solemn", "lonely",
    "peaceful", "serene", "contemplative",
    "playful", "joyful", "whimsical",
    "heroic", "triumphant", "epic",
    "romantic", "intimate", "mysterious", "dreamy", "surreal"]
_SCENE_FRAMING = [_SCENE_NONE,
    # Framing concepts from photography composition; the few in Danbooru's
    # composition tag group (negative space, symmetry, letterboxed) are
    # included but most of these stay as photography vocabulary.
    "centered", "rule of thirds", "leading lines",
    "symmetrical", "asymmetrical",
    "frame within a frame", "negative space",
    "letterboxed", "pillarboxed", "out of frame",
    "dynamic diagonal", "low horizon", "high horizon",
    "shallow depth of field", "deep depth of field"]


class PromptLibraryScene:
    """Structured-form scene description with dropdowns for the common axes.
    Each field's '(none)' option is skipped, so you only contribute the
    attributes you care about. The free-form `extra` textarea catches
    anything the dropdowns don't cover. Use as the 'scene' anchor wired
    into the Comic Frame node so atmosphere stays consistent across panels.
    """

    DESCRIPTION = (
        "Structured scene/atmosphere builder for comic strips. Pick from curated "
        "dropdowns (time of day, weather, lighting, camera angle, mood, framing) "
        "and add anything else in the free-form 'extra' textarea. (none) entries "
        "are skipped. Wire the output into Comic Frame.scene so atmosphere stays "
        "consistent across panels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "time_of_day": (_SCENE_TIME_OF_DAY, {
                    "tooltip": "When the scene is set. Affects sky color, shadow direction, light temperature."}),
                "weather": (_SCENE_WEATHER, {
                    "tooltip": "Atmospheric conditions. Drives haze, reflections, mood."}),
                "lighting": (_SCENE_LIGHTING, {
                    "tooltip": "Light source/quality. Mix with time_of_day for the full atmosphere."}),
                "camera_angle": (_SCENE_CAMERA_ANGLE, {
                    "tooltip": "Camera perspective and shot distance — close-up vs wide, low vs high."}),
                "mood": (_SCENE_MOOD, {
                    "tooltip": "Emotional tone. Influences color grading, expression cues."}),
                "framing": (_SCENE_FRAMING, {
                    "tooltip": "Composition rule — how the subject sits in the frame."}),
                "extra": ("STRING", {"default": "", "multiline": True,
                                      "placeholder": "anything the dropdowns don't cover: lens, "
                                                     "art style, composition refs...",
                    "tooltip": "Free-form extra description appended to the joined scene. "
                               "Use for lens (50mm, anamorphic), art style refs, or anything else."}),
                "separator": ("STRING", {"default": ", ", "multiline": False,
                    "tooltip": "Glue between non-empty fields."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene",)
    OUTPUT_TOOLTIPS = ("All non-skip fields joined by separator. Wire into Comic Frame.scene.",)
    FUNCTION = "build"
    CATEGORY = "GrimmRibbity/Comic"

    def build(self, time_of_day, weather, lighting, camera_angle, mood, framing,
              extra, separator=", "):
        raw = (time_of_day, weather, lighting, camera_angle, mood, framing, extra)
        parts: list[str] = []
        for value in raw:
            if not value:
                continue
            text = str(value).strip()
            if not text or text == _SCENE_NONE:
                continue
            parts.append(text)
        if not parts:
            # All knobs at "(none)" + empty extra — silently emitting "" is
            # confusing when the user expected scene context. Log so the
            # workflow author can tell the Scene node ran but contributed
            # nothing to the prompt.
            print("[PromptLibraryScene] all knobs are (none) and extra is empty — "
                  "emitting empty string. Set a knob (or extra) to contribute scene "
                  "context.")
        return (separator.join(parts),)


_BG_NONE = "(none)"
# Visual divider character. Entries that start with this are treated as
# group headers in the dropdown — selecting one resolves to "(none)" so
# the prompt stays clean even if the user clicks a header by accident.
_BG_DIVIDER_CHAR = "─"
def _bg_divider(label: str) -> str:
    return f"───── {label} ─────"

# Curated background presets, grouped by setting with visual divider rows
# between groups so the dropdown is scannable. Each prompt entry is
# prefixed with the group name so type-to-filter in the COMBO popup
# narrows nicely. Add new entries inside the matching group block.
_BG_PRESETS = [_BG_NONE,
    # NOTE on writing presets: each entry should pin down the LOCATION'S
    # PERSISTENT IDENTITY — wall material/color, floor, ceiling, fixed
    # furniture, fixtures, signage, decor — not time-of-day, weather, or
    # mood (the Scene node handles those). The more specific the anchor,
    # the more consistent the renders across panels of the same comic.
    # ── city / urban ──
    _bg_divider("city / urban"),
    "city: Tokyo street, narrow road wet with rain, neon kanji signs in pink and blue stacked vertically on buildings, vending machines along the sidewalk, tangled overhead power lines, asphalt reflecting the signs",
    "city: New York intersection, wide crosswalks, yellow taxi cabs, steam rising from manhole covers, brownstone facades on the corners, traffic lights on cantilever arms, pedestrian don't-walk hand glowing red",
    "city: European cobblestone alley, narrow stone-paved street between three-story plaster buildings with wooden shutters, wrought-iron balconies, small cafe tables and wicker chairs along one wall, hanging plants",
    "city: rooftop, gravel-covered flat roof, low brick parapet wall, HVAC units and satellite dishes, surrounded by taller skyscrapers with glass curtain walls",
    "city: subway platform, white-tiled walls with a colored line stripe, yellow safety strip along the edge, steel I-beam pillars, hanging fluorescent fixtures, illuminated route sign",
    "city: multi-level parking garage, low concrete ceiling with exposed sprinkler pipes, painted yellow line markings on the floor, square concrete support pillars, ramps with chevron arrows",
    "city: outdoor street market, fabric awnings in bright colors stretched between metal poles, wooden produce crates piled with goods, hand-painted price signs, brick storefronts behind",
    "city: abandoned alleyway, narrow gap between two brick buildings, dumpsters with lids askew, fire escape ladders bolted to the walls, graffiti tags layered on the bricks, trash and broken pallets on the ground",
    "city: fire escape, black-painted steel grating platform bolted to a red-brick tenement, vertical ladder leading up, sash windows with peeling paint, drying laundry on a line",
    "city: gas station forecourt, flat canopy with fluorescent strip lighting, four fuel pumps with dangling hoses, painted lane lines on the concrete, illuminated brand sign on a tall pylon",
    "city: 24-hour convenience store interior, six aisles of shelving stocked with snacks, glass-doored coolers along the back wall, magazine rack near the entrance, fluorescent ceiling tiles, vinyl floor",
    "city: empty bus stop, glass-walled shelter with a metal frame, single steel-and-wood bench inside, route map poster in a backlit frame, painted curb extension",
    "city: parking lot, asphalt with painted white parking lines, scattered cars between the lines, tall sodium vapor lamp on a pole, low concrete bumper blocks, chain-link fence at the back",
    "city: pharmacy aisle, white linoleum floor, tall metal shelving on both sides stocked with boxed medicines, overhead fluorescent panels, category signs hanging from the ceiling, end-cap displays",
    # ── home / interior ──
    _bg_divider("home / interior"),
    "home: cozy bedroom, queen bed with rumpled white duvet and patterned pillows, warm string lights draped along the headboard, hardwood floor with a sheepskin rug, IKEA-style nightstand with a stack of paperbacks, framed art prints on the wall",
    "home: master bedroom, four-poster wood bed with cream linens, plush wool rug centered on the floor, two nightstands flanking the bed, tall window with sheer curtains, dresser with a tilting mirror",
    "home: kid's bedroom, twin bed with cartoon-print bedspread, plastic toy bins stacked in the corner, glow-in-the-dark star stickers on the ceiling, small desk with crayons, posters on the walls",
    "home: teen's bedroom, unmade bed against one wall, band posters and movie posters covering the walls, a desk with a gaming monitor and RGB keyboard, scattered clothes on the floor, fairy lights",
    "home: modern kitchen, white shaker cabinets, white marble countertop with grey veining, stainless steel appliances, herringbone tile backsplash, central island with three stools, pendant lights overhead",
    "home: rustic farmhouse kitchen, exposed wood ceiling beams, butcher-block counters, copper pots hanging from a ceiling rack, white apron-front sink under a window, patterned ceramic tile floor",
    "home: suburban living room, beige microfiber couch and matching loveseat, glass coffee table on a neutral rug, beige walls with framed family photos, large flat-screen TV on a console, ceiling fan",
    "home: minimalist living room, low-profile grey couch, single tall houseplant in a ceramic pot, blank white walls, light oak floor, low matte-black coffee table, no clutter",
    "home: dining room, polished walnut dining table with eight upholstered chairs, brass chandelier with candle-shaped bulbs, sideboard with china displayed, wainscoted walls, oriental area rug",
    "home: study, two leather wingback chairs facing a small fireplace with a wood mantel, built-in dark-wood bookshelves filled with hardcovers, oriental rug, brass desk lamp",
    "home: home library, floor-to-ceiling dark wood bookshelves filled with hardcovers, rolling library ladder on a brass rail, leather armchair with a side table and a green banker's lamp, parquet floor",
    "home: foyer, dark wood floor with a small entry rug, coat rack with hooks, umbrella stand by the door, console table with a mirror above it, framed art on the wall",
    "home: walk-in closet, two facing walls of cedar shelves and hanging rods filled with clothing, central island with drawers, full-length mirror at one end, soft recessed lights",
    "home: laundry room, white front-loading washer and dryer side by side under a counter, open shelving above with detergent and folded towels, sink basin, vinyl tile floor",
    "home: pantry, narrow walk-in space with white shelves on three walls, glass jars of bulk goods labeled by hand, rows of canned goods, single bare bulb",
    "home: messy bathroom, white subway tile walls, beige tile floor, plastic shower curtain partly drawn, vanity counter cluttered with toothbrushes and lotion bottles, fogged-up mirror",
    "home: modern bathroom, freestanding white soaking tub, large-format marble tiles on floor and walls, glass-walled walk-in shower, floating vanity with vessel sink, brushed nickel fixtures",
    "home: cluttered attic, exposed wood rafters and pink fiberglass insulation, dusty cardboard boxes stacked unevenly, an old wardrobe and rocking chair under a single hanging bulb, wood plank floor",
    "home: dim basement, exposed concrete walls, copper and PVC pipes running across the ceiling, washing machine and dryer along one wall, single utility sink, painted concrete floor",
    "home: garage interior, polished concrete floor with oil stains, pegboard wall with hand tools hanging in outlines, metal workbench, rolling toolbox, single-bay sectional door",
    "home: basement workshop, plywood-topped workbench against a wall, table saw in the center, pegboard with hand tools, sawdust drifted in the corners, fluorescent shop light overhead",
    "home: backyard patio, weathered cedar deck with a string of bulb lights overhead, weathered teak outdoor furniture with cream cushions, potted plants along the railing, lawn beyond",
    "home: front porch, painted-wood porch with white railing, two wood rocking chairs, hanging fern, screen door with a wood frame, mailbox by the door",
    "home: small balcony, narrow tile-floored balcony with a wrought-iron railing, two potted plants and a folding bistro table, view of city rooftops beyond",
    "home: greenhouse, glass walls and pitched glass roof, condensation streaking the panes, wooden potting bench, rows of seedling trays under grow lights, hose coiled on the floor",
    "home: sunroom, wall of tall windows on two sides, white wicker chairs with cream cushions, ceramic-tile floor, several large potted palms, ceiling fan",
    "home: spiral staircase, wrought-iron staircase coiling down, hardwood treads, banister with twisted spindles, wall-mounted sconce light",
    # ── hallways / corridors ──
    _bg_divider("hallways / corridors"),
    "hallway: residential hallway, cream walls with chair-rail trim, dark wood baseboards, three white panel doors on each side with brass knobs, beige patterned runner rug down the center, framed family photos in a row, single ceiling sconce",
    "hallway: hotel corridor, identical numbered doors of dark wood every few feet, patterned red-and-gold carpet, wall sconces between doors, recessed ceiling lights, exit sign at the end",
    "hallway: mansion hallway, dark wood paneling waist-high with patterned wallpaper above, marble-tile floor with a long oriental runner, gilt-framed oil portraits on the walls, brass wall sconces, coffered ceiling",
    "hallway: modern apartment hallway, white painted walls, light grey carpet, recessed LED ceiling lights every few feet, identical white doors with chrome lever handles, no decoration",
    "hallway: victorian hallway, busy floral wallpaper, dark stained-wood floorboards with a runner rug, gas-style sconces converted to electric, a stained-glass window at the far end, dado rail",
    "hallway: school hallway, light-blue lockers in continuous rows along both walls, beige speckled linoleum tile floor, fluorescent ceiling fixtures, motivational posters taped to lockers, water fountain",
    "hallway: hospital corridor, white tiled floor with a colored guideline stripe, pale-blue painted walls, steel handrails along both sides, recessed fluorescent lighting, gurneys parked against one wall, signage above doorways",
    "hallway: office building corridor, beige low-pile carpet, beige walls with rubber baseboards, white drop-ceiling tiles with recessed fluorescent panels, wood-veneer doors with placards, water cooler",
    "hallway: prison corridor, barred cell doors on both sides, painted concrete floor with a yellow safety stripe, harsh fluorescent strip lights, painted cinder-block walls, surveillance cameras at intervals",
    "hallway: dungeon stone corridor, mortared stone-block walls and arched ceiling, uneven flagstone floor, iron-banded wood doors set in alcoves, lit torches in iron sconces, damp puddles",
    "hallway: spaceship corridor, white-and-grey wall paneling with riveted seams, blue LED accent strips along the floor, oval bulkhead doors, ribbed ceiling with embedded utility lines",
    "hallway: arcade corridor, dark carpet with neon-pattern, walls of game cabinet marquees glowing in pink and cyan, blacklight strips along the ceiling, hanging neon signage",
    # ── workspaces / studios ──
    _bg_divider("workspaces / studios"),
    "office: corporate cubicle, beige half-height fabric partitions, L-shaped desk with two computer monitors, ergonomic mesh chair, papers and sticky notes pinned to the partition, photo and small plant on the desk",
    "office: open-plan workspace, long shared bench-style desks with monitors, mesh task chairs, glass-walled meeting rooms in the background, whiteboards on wheels, polished concrete floor, exposed ceiling ducts",
    "office: corner office, floor-to-ceiling windows on two walls overlooking a city skyline, executive desk in dark wood, two visitor chairs, area rug, leather couch against the inner wall",
    "office: server room, two rows of black server racks with green and red status LEDs, raised perforated floor tiles, dropped ceiling with cable trays, blue accent lighting, glass entry door",
    "studio: photo studio, white seamless paper backdrop curving down to the floor, two softboxes on stands flanking the set, polished concrete floor with cable-management tape, c-stands and reflectors",
    "studio: recording studio, large mixing console with hundreds of faders, three large studio monitors at the front, acoustic foam panels covering the walls in geometric pattern, glass window into the live room",
    "studio: dance studio, full-wall mirrors on one side, ballet barre running along the mirror, sprung wood floor, exposed ductwork ceiling, large casement windows on the opposite wall",
    "studio: art studio, two wooden easels with canvases in progress, paint-splattered concrete floor, jars of brushes and tubes of oil paint on a table, north-facing windows, drop cloths",
    "studio: pottery studio, two electric pottery wheels with metal splash pans, shelves of unfinished bowls in greenware, kiln in the back corner, bags of clay, clay-dusted concrete floor",
    "workshop: machinist's shop, metal lathe and mill in the foreground, steel chips piled around the bases, tool chests with many small drawers, fluorescent strip lights, oil-stained concrete floor",
    "workshop: woodworking shop, large workbench with a vise, table saw in the center, pegboard wall with chisels and hand planes hanging in outlines, sawdust on the floor, fluorescent shop lights",
    # ── public / commercial ──
    _bg_divider("public / commercial"),
    "cafe: cozy coffee shop interior, exposed brick wall on one side, mismatched wooden tables and chairs, hanging Edison-bulb pendant lights, chalkboard menu behind the counter, espresso machine on the bar",
    "cafe: bustling chain cafe, beige tile floor, light-wood communal tables, glass display case of pastries on the counter, espresso machine with steam rising, baristas in green aprons",
    "restaurant: dim romantic restaurant, dark wood paneling, white linen tablecloths, single tea-light candle on each table, framed art on the walls, dim pendant lights overhead, wine rack along the wall",
    "restaurant: fast-food joint, fluorescent panel ceiling lights, plastic-laminate tables and bolted-down stools, illuminated menu board behind the counter, tile floor, branded signage in red and yellow",
    "bar: dive bar, neon beer signs glowing on the wall, dark wood bar with vinyl-cushioned stools, pool table in the back room with a hanging billiard light, jukebox by the door, sticky linoleum floor",
    "bar: speakeasy, brass-and-leather barstools, marble bar top, backlit shelves of amber-bottled liquor, tufted leather booths along the walls, art-deco geometric carpet, vintage sconces",
    "nightclub: dance floor, illuminated multi-color floor panels, fog machine haze, laser strobes overhead, raised DJ booth with mixers, packed crowd silhouettes, mirrored bar in the background",
    "bookstore: tall floor-to-ceiling dark-wood shelves stuffed with books, narrow aisle, rolling library ladder, reading nook with a worn leather armchair and floor lamp, woven rug",
    "boutique: high-end clothing store, polished concrete floor, individual garments hung on chrome rails with space between, large round mirrors, single-bulb pendant lights, white walls, marble counter",
    "museum: marble exhibition hall, polished marble floor with inlaid pattern, classical sculptures on plinths, vaulted ceiling with skylights, info plaques on the walls, columned doorway",
    "gallery: minimalist white art gallery, white walls and polished concrete floor, large framed canvases at eye level with track-lit spotlights, single white pedestal with a sculpture, no other furniture",
    "theater: empty theater interior, rows of red-velvet folding seats descending toward a stage, heavy red-velvet curtain drawn closed across the proscenium, gilt molding around the stage, low ambient lighting",
    "arcade: arcade hall, rows of game cabinets with glowing marquees, dark patterned carpet, blacklight strips along the ceiling, change machine on the wall, prize counter at the back",
    "salon: hair salon, four styling chairs facing wall-mounted mirrors with light bars, washing station with reclining chair and bowl in the back, white tile floor, product shelves, hairdryers in holders",
    "barbershop: vintage barbershop, two leather-and-chrome barber chairs facing wall mirrors, black-and-white checkered tile floor, rotating barber pole visible through the front window, framed photos",
    "supermarket: long fluorescent-lit aisles, polished beige tile floor, metal shelving stocked floor-to-ceiling, suspended aisle-number signs, end-cap promotional displays, drop-ceiling tiles",
    # ── institutional ──
    _bg_divider("institutional"),
    "school: empty classroom, rows of single-pupil wooden desks with attached metal-frame chairs, large green chalkboard along one wall, teacher's desk at the front, pull-down world map, motivational posters, fluorescent ceiling lights, linoleum floor",
    "school: gymnasium, polished hardwood floor with painted basketball court lines, two retracted basketball hoops on opposite walls, bleachers on one side, exposed steel-truss ceiling, fluorescent shop lights",
    "school: library, rows of low wooden bookshelves, large rectangular tables with green-shaded reading lamps, card catalog cabinet, librarian's desk with a computer, ceiling fans, beige carpet",
    "school: cafeteria, long folding tables with attached plastic stools, beige speckled linoleum floor, lunch counter with sneeze guard along one wall, fluorescent ceiling, signs for daily specials",
    "hospital: corridor, white-tile floor with a colored stripe along one wall, pale-blue painted walls, steel handrails along both sides, gurneys parked against the wall, signage above each doorway, recessed fluorescent lights",
    "hospital: private patient room, single hospital bed with adjustable rails and IV pole beside it, vinyl-floor, beige walls, vinyl-upholstered visitor chair, wall-mounted television on a swing arm, blinds on the window",
    "hospital: operating theater, central operating table under a multi-armed surgical light, sterile blue drapes, anesthesia cart and monitors on stands, glass-fronted supply cabinets, polished tile floor",
    "hospital: emergency room bay, single gurney behind a privacy curtain on a track, monitor cart, IV pole, suction equipment on the wall, biohazard bin, vinyl floor, glass doors at the entry",
    "courtroom: dark walnut paneling, judge's bench raised on a platform with the seal mounted behind, jury box of twelve seats on one side, two counsel tables facing the bench, gallery seating for spectators, wood gavel on the bench",
    "police: interrogation room, bare painted-cinder-block walls, single rectangular metal table bolted to the floor with two metal chairs, two-way mirror on one wall, fluorescent fixture overhead, polished concrete floor",
    "police: precinct bullpen, rows of cluttered detective desks with monitors and stacks of files, corkboards with photos and red-string connections, water cooler, holding cell visible at the back, fluorescent ceiling",
    "therapist: therapy office, leather chesterfield couch along one wall, leather wingback chair facing it, low coffee table with tissues and a clock, bookshelves filled with hardcover psychology books, oriental rug, soft floor lamp",
    "prison: cell, painted cinder-block walls, single bunk bolted to the wall with a thin mattress, stainless-steel sink-toilet combo unit, small barred window high on the back wall, polished concrete floor, hinged steel door",
    "morgue: cold storage room, wall of stainless-steel body-storage drawers, two stainless-steel autopsy tables in the center with overhead lights, tile floor with central drains, hanging surgical lights",
    # ── nature / outdoor ──
    _bg_divider("nature / outdoor"),
    "nature: forest clearing, ring of tall pine trees surrounding a circular grassy clearing, scattered fallen logs, mossy rocks, a stream visible through the trees on one side, ferns at the edge",
    "nature: mountain peak, rocky summit with patches of snow in the crevices, sweeping view of lower peaks below the cloud line, weathered stone cairn at the top, alpine grasses",
    "nature: ocean cliff, sheer rock cliff edge with grass at the top, white-capped waves crashing on rocks far below, seabirds wheeling, distant horizon line",
    "nature: riverbank, slow-moving river bordered by reeds and tall grasses, smooth round skipping stones at the water's edge, gravel bank, willow trees overhanging from the far side",
    "nature: autumn forest, deciduous trees with orange and red leaves, thick carpet of fallen leaves on the ground, narrow dirt trail winding through, fallen log covered in moss",
    "nature: snowy field, flat expanse of pristine snow with a single trail of footprints crossing it, low rolling hills in the distance, bare-branched trees against the sky, no other features",
    "nature: beach, broad expanse of fine sand at low tide showing ripple patterns and shells, tide line of seaweed and driftwood, calm water, low-tide dunes with sea grass",
    "nature: desert dunes, rolling expanse of pale sand dunes with wind-rippled surfaces, sharp ridge lines, no vegetation in sight, deep blue sky",
    "nature: dense jungle, thick tropical undergrowth, hanging vines and lianas, broad-leaved palms, dappled green light filtering through the canopy, exposed roots, mossy rocks",
    "nature: cave entrance, rocky cave mouth seen from inside, daylight streaming in framed by mossy boulders and overgrown ferns, small puddle reflecting the entrance, dripping stalactites near the opening",
    "nature: alpine meadow, rolling grassy slope dotted with wildflowers in patches of yellow and purple, low scattered rocks, mountain ridge in the distance, single twisted tree",
    "nature: pine forest, evenly spaced tall pine trunks rising into a dense canopy, pine-needle carpet on the floor, mist hanging between the trunks, narrow shafts of light",
    # ── industrial / decay ──
    _bg_divider("industrial / decay"),
    "industrial: abandoned warehouse, vast open floor space, broken multi-pane factory windows high on the walls letting in shafts of light, scattered wooden crates and pallets, exposed steel-truss ceiling, oil-stained concrete floor, weeds growing in the cracks",
    "industrial: factory floor, conveyor belts running between heavy stamping machines, exposed industrial ductwork overhead, painted yellow safety lines on the polished concrete floor, sodium-vapor lamps, control panels along one wall",
    "industrial: shipping yard, stacked rows of color-faded shipping containers, gantry cranes overhead, asphalt service roads between the rows, container numbers stenciled in white",
    "industrial: derelict subway tunnel, twin rusted steel rails on a debris-strewn track bed, water dripping from cracks in the curved tunnel wall, third rail visible, single working bulb in a wire cage casting weak light",
    "industrial: construction site, partial steel skeleton of a high-rise visible in the background, scaffolding wrapped in green safety mesh, exposed rebar, stacked construction materials, dirt and gravel ground",
    "industrial: power plant turbine hall, massive cylindrical turbines along the floor, gantry crane on tracks overhead, control catwalks running along the walls, polished concrete floor, instrument panels",
    "industrial: junkyard, mountains of stacked car wrecks and crumpled body panels, dirt access roads weaving between, chain-link perimeter fence with barbed wire, single excavator with a magnet attachment",
    # ── fantasy ──
    _bg_divider("fantasy"),
    "fantasy: medieval castle great hall, soaring vaulted stone ceiling with wooden beams, banners hanging from the rafters, two long trestle tables down the length, large stone fireplace at one end with a fire burning, iron sconces with torches",
    "fantasy: wizard's tower study, circular stone room, wraparound bookshelves filled with leather tomes, a wooden desk strewn with rolled scrolls, alchemical glassware on a side table, narrow arrow-slit windows, an astrolabe in the corner",
    "fantasy: tavern interior, low-beamed wood ceiling, long rectangular wooden bar along one wall with mugs hanging above, scattered round wooden tables and stools, large stone fireplace with a fire crackling, plank floor with rushes",
    "fantasy: enchanted forest, ancient gnarled trees with thick trunks, carpet of glowing blue mushrooms at their bases, hanging paper lanterns swinging on cords between branches, mossy rocks, a stream running through",
    "fantasy: throne room, two rows of stone columns flanking a central aisle, raised stone dais at the far end with an ornate stone throne, banners hanging behind the throne, polished flagstone floor, vaulted ceiling",
    "fantasy: dungeon cell, small stone-block room, iron rings set in the walls with hanging chains, a single sconce holding a torch, rusted iron-banded wood door with a barred slot, straw scattered on the flagstone floor",
    "fantasy: blacksmith's forge, central brick forge with glowing coals, cast-iron anvil mounted on a tree-trunk stump beside it, wall-mounted rack of hammers and tongs, water-cooling barrel, bellows on a chain",
    "fantasy: alchemist's laboratory, long wooden workbench covered with bubbling flasks and beakers connected by glass tubes, dried herbs hanging in bunches from the ceiling, mortar and pestle, leather-bound book open on a stand",
    # ── sci-fi ──
    _bg_divider("sci-fi"),
    "scifi: spaceship bridge, captain's chair on a raised platform in the center, semicircle of crew workstations with curved holographic displays, large viewport at the front showing stars and a planet's edge, blue accent lighting along the floor edges",
    "scifi: alien planet surface, expanse of purple sand under a violet sky, twin suns low on the horizon, jagged rock formations in unusual angular shapes, scattered crystalline outcroppings",
    "scifi: cyberpunk neon street, narrow rain-slick city street, towering buildings with multi-story holographic billboards in pink and cyan, hanging power lines, food stalls with backlit signs along one side, puddles reflecting the lights",
    "scifi: post-apocalyptic ruins, half-collapsed skyscraper skeletons overgrown with vines, vehicles overturned and rusted on a cracked highway, ash-coated ground, dust haze, dead trees",
    "scifi: clean white laboratory, gleaming white floor and walls, glass partition walls between workstations, chrome equipment racks holding scientific instruments, recessed LED lighting",
    "scifi: space station observation deck, curved floor-to-ceiling viewport along one wall showing a planet rotating below, low metal benches facing the view, indirect blue lighting, polished metal floor",
    "scifi: cryo-chamber room, two facing rows of vertical cryo-pods with frosted glass fronts, internal blue glow visible through the frost, corrugated metal floor, control consoles between the pod rows",
    "scifi: mech bay, towering humanoid mech standing on a service platform, gantry catwalks at multiple heights for technicians, hanging tool arms on cables, pooled hydraulic fluid on the floor, harsh work-light beams"]


class PromptLibraryBackground:
    """Background anchor for a comic. Pick a preset from the curated list,
    add custom detail in the textarea (both contribute when set), or skip
    the preset entirely and just type your own. Wire the output into the
    Comic Frame node so the location reads as continuous across panels.
    """

    DESCRIPTION = (
        "Locked background description for a comic strip. Pick a curated preset "
        "(grouped by setting: city / home / office / nature / industrial / "
        "fantasy / scifi / etc.) and/or add free-form custom detail. Both "
        "contribute when set. Wire the output into Comic Frame.background — "
        "the location reads as continuous across all panels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (_BG_PRESETS, {
                    "tooltip": "Curated background description. (none) skips the preset entirely "
                               "so you can rely on 'custom' alone."}),
                "custom": ("STRING", {"default": "", "multiline": True,
                                       "placeholder": "extra detail, time-of-day overrides, "
                                                      "props specific to this scene...",
                    "tooltip": "Free-form text appended after the preset. Either field alone "
                               "works; both together get joined with the separator."}),
                "separator": ("STRING", {"default": ", ", "multiline": False,
                    "tooltip": "Glue between preset and custom when both are set."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("background",)
    OUTPUT_TOOLTIPS = ("Final background description. Wire into Comic Frame.background.",)
    FUNCTION = "build"
    CATEGORY = "GrimmRibbity/Comic"

    def build(self, preset, custom, separator=", "):
        parts: list[str] = []
        # Skip the (none) sentinel and any visual divider row the user picked
        # by accident — divider entries start with the divider character.
        preset_str = (str(preset) if preset is not None else "").strip()
        if preset_str and preset_str != _BG_NONE and not preset_str.startswith(_BG_DIVIDER_CHAR):
            parts.append(preset_str)
        custom = (custom or "").strip()
        if custom:
            parts.append(custom)
        return (separator.join(parts),)


class PromptLibraryComicFrame:
    """Comic-strip frame assembler. Combines anchor STRINGs (character / scene
    / background) with the per-frame action text for the panel selected by
    `frame_index`. Frames are authored in the node's UI as an ordered list
    of textareas; the list is persisted into the workflow JSON.
    """

    DESCRIPTION = (
        "Comic-strip frame assembler. Combines anchor STRINGs (character / scene / "
        "background) with the per-frame action selected by frame_index. Frames are "
        "authored in the node's UI as an ordered list of textareas; the list is "
        "saved with the workflow. Bump frame_index between queues to render each "
        "panel — anchors stay locked so the comic reads as continuous."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # JSON list of frame action strings, edited via the DOM widget
                # in the frontend. Stored on the workflow so it round-trips.
                "frames_json": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "JSON array of per-frame action texts. Driven by the frame "
                               "editor widget in the node — you don't normally type here."}),
                "frame_index": ("INT", {"default": 1, "min": 1, "max": 999,
                    "tooltip": "1-based index of the panel to render. Auto-clamped to the "
                               "frame count. Bump between queues to render each panel."}),
                "separator": ("STRING", {"default": ", ", "multiline": False,
                    "tooltip": "Glue between character / scene / background / action."}),
            },
            "optional": {
                "character": ("STRING", {"default": "", "multiline": True,
                                          "forceInput": True,
                    "tooltip": "Wire from a Library node filtered to your character entries. "
                               "Repeated verbatim on every frame."}),
                "scene": ("STRING", {"default": "", "multiline": True,
                                      "forceInput": True,
                    "tooltip": "Wire from a GrimmRibbity Scene node. Repeated verbatim on every frame."}),
                "background": ("STRING", {"default": "", "multiline": True,
                                           "forceInput": True,
                    "tooltip": "Wire from a GrimmRibbity Background node. Repeated verbatim on every frame."}),
                # Negative-side wires — flow Library's `negative` output into
                # any of these and ComicFrame joins them into one negative
                # output so a single sampler pair (positive/negative) sees the
                # full per-frame stack.
                "character_negative": ("STRING", {"default": "", "multiline": True,
                                                   "forceInput": True,
                    "tooltip": "Optional. Wire from a Library node's `negative` output for the "
                               "character. Repeated verbatim on every frame."}),
                "scene_negative": ("STRING", {"default": "", "multiline": True,
                                               "forceInput": True,
                    "tooltip": "Optional. Negative-prompt counterpart for the scene wire."}),
                "background_negative": ("STRING", {"default": "", "multiline": True,
                                                    "forceInput": True,
                    "tooltip": "Optional. Negative-prompt counterpart for the background wire."}),
                # Adds frame_index to whatever upstream seed you wire in,
                # giving each panel a deterministic-but-different seed.
                "base_seed": ("INT", {"default": 0, "min": 0, "max": _INT_MAX,
                    "tooltip": "Base for the per-frame seed output. Frame N emits "
                               "base_seed + (N - 1). Wire the seed output into "
                               "CivitaiSaveImage.seed_override for deterministic-but-different "
                               "seeds across panels."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "action", "seed", "frame_count", "negative")
    OUTPUT_TOOLTIPS = (
        "Final positive prompt: character + scene + background + frames[index].",
        "Just the per-frame action text. Useful for filename builders or text overlays.",
        "base_seed + (frame_index - 1). Wire into CivitaiSaveImage.seed_override.",
        "Total number of authored frames. Useful for downstream branching/looping.",
        "Joined negative prompt from any wired *_negative inputs (empty if none).",
    )
    FUNCTION = "assemble"
    CATEGORY = "GrimmRibbity/Comic"

    @staticmethod
    def _parse_frames(frames_json: str) -> list[str]:
        try:
            data = json.loads(frames_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        return [str(f).strip() for f in data]

    def assemble(self, frames_json, frame_index, separator=", ",
                 character="", scene="", background="",
                 character_negative="", scene_negative="", background_negative="",
                 base_seed=0):
        frames = self._parse_frames(frames_json)
        if not frames:
            print("[PromptLibraryComicFrame] no frames defined; emitting anchors only")
            action = ""
        else:
            # Clamp into range and 1-index for the user-visible widget.
            idx = max(1, min(int(frame_index), len(frames))) - 1
            action = frames[idx]
        parts = [p.strip() for p in (character, scene, background, action)
                 if p and p.strip()]
        prompt = separator.join(parts)
        neg_parts = [p.strip() for p in (character_negative, scene_negative, background_negative)
                      if p and p.strip()]
        negative = separator.join(neg_parts)
        seed = (int(base_seed) + max(0, int(frame_index) - 1)) & _INT_MAX
        return (prompt, action, seed, len(frames), negative)


if PromptServer is not None:
    routes = PromptServer.instance.routes
else:
    # Standalone import (maintenance tools): a detached RouteTableDef silently
    # collects the @routes decorators without ever being attached to an app.
    routes = web.RouteTableDef()


async def _json_payload(request) -> tuple[dict, web.Response | None]:
    """Read a JSON POST body, returning (payload, error_response_or_None).

    Centralises the pattern that was inconsistent across routes: some
    routes called `await request.json()` raw (raises 500 on a malformed
    body), some checked `request.body_exists` first. The standard now is:
    no body → empty dict (no error); malformed JSON → 400 with a clear
    message. Callers either get a usable dict or a Response to return
    directly."""
    if not request.body_exists:
        return {}, None
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError) as e:
        return {}, web.json_response(
            {"error": f"invalid JSON body: {e}"}, status=400)
    if not isinstance(payload, dict):
        return {}, web.json_response(
            {"error": "request body must be a JSON object"}, status=400)
    return payload, None


def _scan_image_ids() -> set[str]:
    """Build a set of every prompt_id that has a thumbnail file in
    IMAGES_DIR. One os.scandir replaces N × (extensions) Path.exists()
    syscalls. Only used by the /list route — the per-id _image_path_for
    is fine for the 1-2 callers that actually need the path itself.

    For a 700-entry library, this turns 700 * 7 = 4900 stat() syscalls
    per /list response into a single readdir(). Critical when the
    gallery refreshes every websocket event."""
    valid_exts = _ALLOWED_IMAGE_EXT
    found: set[str] = set()
    try:
        with os.scandir(IMAGES_DIR) as it:
            for entry in it:
                name = entry.name
                dot = name.rfind(".")
                if dot < 1:
                    continue
                if name[dot:].lower() not in valid_exts:
                    continue
                found.add(name[:dot])
    except OSError:
        pass
    return found


@routes.get("/prompt_library/list")
async def list_prompts(_request):
    with _lock:
        items = _load()
    image_ids = _scan_image_ids()
    out = []
    for idx, item in enumerate(items):
        pid = item.get("id", "")
        out.append({
            "id": pid,
            "name": item.get("name", ""),
            "text": item.get("text", ""),
            "negative": item.get("negative", ""),
            "tags": item.get("tags", []),
            "rating": int(item.get("rating", 0) or 0),
            "notes": item.get("notes", ""),
            "loras": item.get("loras", []),
            "created_at": item.get("created_at", 0),
            "updated_at": item.get("updated_at", 0),
            "order": item.get("order", idx),
            "has_image": pid in image_ids,
        })
    return web.json_response({"prompts": out})


@routes.get("/prompt_library/loras")
async def list_loras(_request):
    """Names of every .safetensors LoRA visible to ComfyUI's folder_paths.
    Sourced from get_filename_list("loras") so subfolders are preserved
    (e.g. 'Anima/Anima Turbo LoRA.safetensors'). Frontend uses this to
    populate the modal's LoRA-row dropdown."""
    if folder_paths is None:
        return web.json_response({"loras": [], "error": "folder_paths unavailable"})
    try:
        names = list(folder_paths.get_filename_list("loras") or [])
    except Exception as e:
        return web.json_response({"loras": [], "error": str(e)})
    return web.json_response({"loras": sorted(names)})


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
    negative = reader.get("negative") or ""
    tags = _parse_tags(reader.get("tags"))
    notes = (reader.get("notes") or "").strip()
    loras_field = reader.get("loras")
    # Distinguish "loras key omitted" (older clients, don't touch the field)
    # from "loras key present but empty" (cleared in the modal, write []).
    loras_provided = "loras" in reader
    loras = _parse_loras(loras_field) if loras_provided else None
    rating_raw = reader.get("rating")
    rating: int | None = None
    if rating_raw not in (None, ""):
        try:
            rating = max(0, min(5, int(rating_raw)))
        except (ValueError, TypeError):
            return web.json_response({"error": "rating must be an integer 0-5"}, status=400)
    clear_image = (reader.get("clear_image") or "") == "1"
    image_field = reader.get("image")

    if not name:
        return web.json_response({"error": "name required"}, status=400)
    if pid and not _safe_id(pid):
        return web.json_response({"error": "invalid id"}, status=400)

    # Read + validate image bytes outside the lock so a corrupt upload doesn't
    # leave the entry partially mutated. If validation passes, we hold the
    # decoded bytes and write the thumbnail under the lock; on thumbnail-write
    # failure we roll the entry back to its pre-upsert snapshot.
    pending_image_bytes: bytes | None = None
    if image_field is not None and hasattr(image_field, "file") and image_field.filename:
        ext = os.path.splitext(image_field.filename)[1].lower()
        if ext not in _ALLOWED_IMAGE_EXT:
            return web.json_response({"error": f"unsupported image type {ext}"}, status=400)
        pending_image_bytes = image_field.file.read()
        if len(pending_image_bytes) > _MAX_IMAGE_BYTES:
            return web.json_response({"error": "image too large"}, status=400)

    with _lock:
        items = _load()
        if not pid:
            pid = _unique_id(_slugify(name), {i.get("id") for i in items})
        existing = next((i for i in items if i.get("id") == pid), None)
        created = existing is None
        snapshot = None if created else dict(existing)
        if created:
            existing = {"id": pid}
            items.append(existing)
        else:
            _maybe_push_history(existing, name, text, tags, new_negative=negative,
                                  new_loras=loras if loras_provided else existing.get("loras"))
        existing["name"] = name
        existing["text"] = text
        existing["tags"] = tags
        if negative:
            existing["negative"] = negative
        elif "negative" in existing:
            existing["negative"] = ""
        if notes:
            existing["notes"] = notes
        elif "notes" in existing:
            existing["notes"] = ""
        if rating is not None:
            existing["rating"] = rating
        if loras_provided:
            if loras:
                existing["loras"] = loras
            elif "loras" in existing:
                existing["loras"] = []
        _touch(existing, created=created)

        if clear_image:
            _delete_image_files(pid)

        if pending_image_bytes is not None:
            if _save_thumbnail_bytes(pid, pending_image_bytes) is None:
                # Roll back the entry mutation so we don't half-commit.
                if created:
                    items.remove(existing)
                else:
                    existing.clear()
                    existing.update(snapshot)
                return web.json_response({"error": "failed to process image"}, status=400)

        _save(items)

    _notify_change()
    return web.json_response({
        "id": pid,
        "name": name,
        "text": text,
        "tags": tags,
        "rating": existing.get("rating", 0),
        "notes": existing.get("notes", ""),
        "loras": existing.get("loras", []),
        "has_image": _image_path_for(pid) is not None,
    })


def _import_csv(text: str, *, mode: str = "add_only") -> tuple[int, int, int, list[str]]:
    """Parse CSV body and upsert each row. Returns (added, updated, skipped, errors).

    mode="add_only" (default) — entries whose resolved id already exists in
    the library are skipped, preserving any thumbnail / rating / notes /
    edits the user has made. Only genuinely new entries get added.

    mode="update" — existing entries are overwritten with the CSV row's
    values (the historical default; equivalent to upsert)."""
    if mode not in ("add_only", "update"):
        raise ValueError(f"unknown import mode {mode!r}")
    reader = csv.DictReader(io.StringIO(text))
    added = updated = skipped = 0
    errors: list[str] = []
    if not reader.fieldnames or "name" not in reader.fieldnames:
        return 0, 0, 0, ["CSV missing required 'name' column"]

    _snapshot_prompts("import_csv")
    with _lock:
        items = _load()
        index = {i.get("id"): i for i in items}
        for row_num, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                errors.append(f"row {row_num}: empty name")
                continue
            text_val = row.get("text", "") or ""
            negative_val = (row.get("negative") or "").strip()
            # Tags use ';' inside CSV cell since ',' is the field delimiter.
            tags = _parse_tags((row.get("tags") or "").replace(";", ","))
            row_id = (row.get("id") or "").strip()
            if row_id and not _safe_id(row_id):
                errors.append(f"row {row_num}: invalid id {row_id!r}")
                continue
            if not row_id:
                row_id = _unique_id(_slugify(name), set(index.keys()))

            existing = index.get(row_id)
            if existing is not None and mode == "add_only":
                skipped += 1
                continue
            created = existing is None
            if created:
                existing = {"id": row_id}
                items.append(existing)
                index[row_id] = existing
                added += 1
            else:
                _maybe_push_history(existing, name, text_val, tags,
                                      new_negative=negative_val)
                updated += 1
            existing["name"] = name
            existing["text"] = text_val
            existing["tags"] = tags
            # Same blank-clears-the-field semantics as Save / upsert: only
            # write the field if the row supplies one OR the entry already
            # has one (so a blank cell on an update genuinely clears it).
            if negative_val or "negative" in existing:
                existing["negative"] = negative_val
            _touch(existing, created=created)

        _save(items)

    return added, updated, skipped, errors


@routes.post("/prompt_library/import_csv")
async def import_csv_route(request):
    reader = await request.post()
    field = reader.get("file")
    if field is not None and hasattr(field, "file"):
        raw = field.file.read()
    else:
        raw_str = reader.get("csv") or ""
        raw = raw_str.encode("utf-8") if isinstance(raw_str, str) else b""
    # Size cap before decode + parse — a multi-GB CSV would otherwise
    # parse into Python memory unchecked. Image and zip uploads were
    # already capped; CSV is now consistent with them.
    if len(raw) > _MAX_IMPORT_CSV_BYTES:
        return web.json_response({
            "error": f"CSV too large: {len(raw) // (1024 * 1024)} MB > "
                     f"{_MAX_IMPORT_CSV_BYTES // (1024 * 1024)} MB cap"
        }, status=400)
    body = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if not body.strip():
        return web.json_response({"error": "no CSV body provided"}, status=400)
    mode = (reader.get("mode") or "add_only").strip()
    if mode not in ("add_only", "update"):
        return web.json_response({"error": f"invalid mode {mode!r} (use add_only or update)"}, status=400)
    added, updated, skipped, errors = _import_csv(body, mode=mode)
    _notify_change()
    return web.json_response({"added": added, "updated": updated,
                               "skipped": skipped, "errors": errors})


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
    payload, err = await _json_payload(request)
    if err is not None:
        return err
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

        snap_loras = _parse_loras(snap.get("loras")) if "loras" in snap else None
        # Save the current state so the revert itself is undoable.
        _maybe_push_history(item, snap.get("name", ""), snap.get("text", ""),
                              snap.get("tags") or [],
                              new_loras=snap_loras if snap_loras is not None
                                         else item.get("loras"))
        item["name"] = snap.get("name", "")
        item["text"] = snap.get("text", "")
        item["tags"] = list(snap.get("tags") or [])
        if snap_loras is not None:
            item["loras"] = snap_loras
        _touch(item, created=False)
        _save(items)

    _notify_change()
    return web.json_response({
        "id": pid,
        "name": item["name"],
        "text": item["text"],
        "tags": item["tags"],
    })


def _raise_aiohttp_client_max_size():
    """Bump ComfyUI's aiohttp Application body cap so our import endpoints can
    accept libraries bigger than ComfyUI's --max-upload-size default (100 MB).

    Without this the framework returns 413 before our per-route cap can run,
    even on a 90 MB zip — multipart overhead and Content-Length rounding push
    it over the edge.
    """
    try:
        from server import PromptServer
    except Exception:
        return
    inst = getattr(PromptServer, "instance", None)
    app = getattr(inst, "app", None)
    if app is None:
        return
    cur = getattr(app, "_client_max_size", 0) or 0
    wanted = max(_MAX_IMPORT_ZIP_BYTES, 500 * 1024 * 1024)
    if cur < wanted:
        app._client_max_size = wanted
        print(f"[PromptLibrary] raised aiohttp client_max_size: {cur} -> {wanted}")


_raise_aiohttp_client_max_size()


def _recompress_thumbnail_for_export(img_path: "Path") -> tuple[bytes, str] | None:
    """Re-encode a stored thumbnail at the export-only smaller settings
    (_EXPORT_THUMBNAIL_MAX_EDGE / _EXPORT_THUMBNAIL_JPEG_QUALITY).
    Returns (bytes, ext) or None if PIL can't decode the source.

    This is what shrinks shared library zips from ~250 MB down to ~50 MB
    for typical 4000-entry libraries: the storage settings keep nice
    400 px / q80 thumbnails on disk, but the transport zip ships
    384 px / q75. Re-importing through /import_zip puts them back through
    _save_thumbnail_bytes, which re-saves at the storage settings —
    enlargement is skipped, so the recipient ends up with the same
    smaller thumbnails the export shipped, not a re-bloat back to 400."""
    try:
        from PIL import Image
        pil = Image.open(img_path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil.thumbnail((_EXPORT_THUMBNAIL_MAX_EDGE, _EXPORT_THUMBNAIL_MAX_EDGE))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=_EXPORT_THUMBNAIL_JPEG_QUALITY,
                 optimize=True, progressive=True)
        return buf.getvalue(), "jpg"
    except Exception as e:
        print(f"[PromptLibrary] export recompress failed for {img_path}: {e}")
        return None


def _build_export_zip(items: list[dict], version: str) -> bytes:
    """Bundle prompts + thumbnail images into a zip. Thumbnails are
    re-encoded at the smaller export settings on the way out — matters
    a lot for shared libraries (the user's 4061-entry export went from
    254 MB to 52 MB after this kicked in)."""
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
            if not img_path:
                continue
            recompressed = _recompress_thumbnail_for_export(img_path)
            if recompressed is not None:
                data, ext = recompressed
                zf.writestr(f"images/{pid}.{ext}", data)
            else:
                # PIL couldn't decode — fall back to the on-disk file as-is
                # so the entry still has SOMETHING shipped with it.
                zf.write(img_path, arcname=f"images/{img_path.name}")
    return buf.getvalue()


@routes.post("/prompt_library/export")
async def export_zip(request):
    payload, err = await _json_payload(request)
    if err is not None:
        return err
    requested = payload.get("ids") or []
    if not isinstance(requested, list):
        return web.json_response(
            {"error": "ids must be a list (or omit to export everything)"},
            status=400)
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


def _import_zip(zip_bytes: bytes, *, mode: str = "add_only") -> tuple[int, int, int, list[str]]:
    """Import a GrimmRibbity export zip. Returns (added, updated, skipped, errors).

    mode="add_only" (default) — entries whose id already exists in the
    library are skipped (no overwrite, no history push, no thumbnail
    replace). Protects local edits / curation when re-importing a shared
    zip.

    mode="update" — existing entries are overwritten with the zip's
    values, with history push for user-visible field changes (the
    historical default behaviour)."""
    if mode not in ("add_only", "update"):
        return 0, 0, 0, [f"unknown import mode {mode!r}"]
    errors: list[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return 0, 0, 0, ["not a valid zip file"]
    # Zip-bomb defence: sum the uncompressed sizes from each member's
    # ZipInfo header BEFORE decompressing anything. A zip can be ~500 MB
    # on disk and expand to dozens of GB; without this, _import_zip would
    # happily read every member into memory and OOM the process.
    try:
        total_uncompressed = sum(getattr(info, "file_size", 0) or 0
                                   for info in zf.infolist())
    except Exception:
        total_uncompressed = 0
    if total_uncompressed > _MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES:
        zf.close()
        return 0, 0, 0, [
            f"zip uncompressed size {total_uncompressed // (1024 * 1024)} MB exceeds "
            f"cap {_MAX_IMPORT_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB — refusing "
            f"to decompress (zip-bomb guard)"]

    with zf:
        try:
            manifest_raw = zf.read("prompts.json").decode("utf-8")
            manifest = json.loads(manifest_raw)
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            return 0, 0, 0, [f"missing or invalid prompts.json in zip: {e}"]

        new_prompts = manifest.get("prompts") if isinstance(manifest, dict) else None
        if not isinstance(new_prompts, list):
            return 0, 0, 0, ["prompts.json has no 'prompts' list"]

        added = updated = skipped = 0
        names_in_zip = set(zf.namelist())

        _snapshot_prompts("import_zip")
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

                existing = index.get(pid)
                if existing is not None and mode == "add_only":
                    # Add-only: don't overwrite established entries. Don't
                    # touch the thumbnail either (might've been customised).
                    skipped += 1
                    continue

                text_val = raw.get("text", "") or ""
                tags = _parse_tags(raw.get("tags") or [])
                history = raw.get("history") if isinstance(raw.get("history"), list) else None
                incoming_loras = _parse_loras(raw.get("loras")) if "loras" in raw else None

                created = existing is None
                if created:
                    existing = {"id": pid}
                    items.append(existing)
                    index[pid] = existing
                    added += 1
                else:
                    _maybe_push_history(existing, name, text_val, tags,
                                          new_loras=incoming_loras if incoming_loras is not None
                                                   else existing.get("loras"))
                    updated += 1

                existing["name"] = name
                existing["text"] = text_val
                existing["tags"] = tags
                # Optional editorial fields — only copy if present in the zip,
                # so re-importing a v1 export (no rating/notes) doesn't wipe
                # newer values on an existing entry.
                if "rating" in raw:
                    try:
                        existing["rating"] = max(0, min(5, int(raw.get("rating") or 0)))
                    except (ValueError, TypeError):
                        pass
                if "notes" in raw:
                    existing["notes"] = str(raw.get("notes") or "")
                # The negative field was silently dropped on import in versions
                # prior to v0.40.1 — exports carried it but the import path
                # didn't read it back, leaking data on every round-trip.
                if "negative" in raw:
                    existing["negative"] = str(raw.get("negative") or "")
                if incoming_loras is not None:
                    existing["loras"] = incoming_loras
                if history is not None and not created:
                    # Merge histories (incoming first, then existing); cap.
                    merged = list(history) + list(existing.get("history") or [])
                    existing["history"] = merged[-_HISTORY_CAP:]
                elif history is not None:
                    existing["history"] = list(history)[-_HISTORY_CAP:]
                _touch(existing, created=created)

                # Restore thumbnail if present in zip — downscale on the way in
                # so re-imports of older full-res libraries shrink.
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
                        if _save_thumbnail_bytes(pid, img_data) is None:
                            errors.append(f"image for {pid!r} could not be processed")
                        break

            _save(items)

    return added, updated, skipped, errors


@routes.post("/prompt_library/import_zip")
async def import_zip_route(request):
    reader = await request.post()
    field = reader.get("file")
    if field is None or not hasattr(field, "file"):
        return web.json_response({"error": "no zip file provided"}, status=400)
    body = field.file.read()
    if len(body) > _MAX_IMPORT_ZIP_BYTES:
        return web.json_response({"error": "zip too large"}, status=400)
    mode = (reader.get("mode") or "add_only").strip()
    if mode not in ("add_only", "update"):
        return web.json_response({"error": f"invalid mode {mode!r} (use add_only or update)"}, status=400)
    added, updated, skipped, errors = _import_zip(body, mode=mode)
    _notify_change()
    return web.json_response({"added": added, "updated": updated,
                               "skipped": skipped, "errors": errors})


# --------------------------------------------------------------------------
# LoRA scan / bulk import — turns every installed .safetensors LoRA into a
# library entry with auto-detected preview thumbnail and trigger words from
# the safetensors metadata. Re-runnable: existing entries are skipped or
# refreshed (no duplicate spam) based on the LoRA's relative path.
# --------------------------------------------------------------------------

_LORA_PREVIEW_EXTS = (".preview.png", ".preview.jpg", ".preview.jpeg",
                       ".preview.webp", ".png", ".jpg", ".jpeg", ".webp")
# Hard-cap how many top trigger tags we ingest from a single LoRA. Some
# Kohya training runs record thousands of tags with falling frequencies;
# the long tail isn't useful and would bloat the entry's tag list.
_LORA_MAX_TRIGGER_TAGS = 12
# Per-tag minimum frequency: skip very rare tags that appear once or twice
# across the whole dataset — usually one-off names that aren't real triggers.
_LORA_MIN_TAG_FREQ = 5
_LORA_TAG_NORMALIZE = str.maketrans({"_": " "})


def _find_lora_preview(full_path: str) -> str | None:
    """Look for a preview image stored alongside the .safetensors. Civitai's
    SD-Civitai-Helper writes <name>.preview.png; some Kohya / kohya_ss
    setups write <name>.png. Returns the first match or None."""
    base, _ = os.path.splitext(full_path)
    for ext in _LORA_PREVIEW_EXTS:
        candidate = base + ext
        try:
            if os.path.isfile(candidate):
                return candidate
        except OSError:
            continue
    return None


def _read_safetensors_metadata(full_path: str) -> dict:
    """Parse only the safetensors header so we don't have to load the
    weights. Format: 8-byte little-endian length, then JSON header, then
    tensor data. The training metadata lives under '__metadata__'.
    Returns an empty dict on any failure (corrupt file, missing key, etc.)."""
    import struct
    try:
        with open(full_path, "rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                return {}
            (header_size,) = struct.unpack("<Q", length_bytes)
            # Defensive cap — header should be tiny relative to the file.
            if header_size <= 0 or header_size > 100 * 1024 * 1024:
                return {}
            header_raw = f.read(header_size)
        if len(header_raw) != header_size:
            return {}
        header = json.loads(header_raw.decode("utf-8"))
        meta = header.get("__metadata__")
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _extract_lora_triggers(metadata: dict, top_n: int = _LORA_MAX_TRIGGER_TAGS) -> list[str]:
    """Pull the most common training tags from Kohya's `ss_tag_frequency`,
    summed across all training subfolders. Tags returned in descending
    frequency, capped at top_n, normalised (underscores → spaces)."""
    raw = metadata.get("ss_tag_frequency")
    if not isinstance(raw, str):
        return []
    try:
        freq_by_dir = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(freq_by_dir, dict):
        return []
    summed: dict[str, int] = {}
    for cat in freq_by_dir.values():
        if not isinstance(cat, dict):
            continue
        for tag, count in cat.items():
            if not isinstance(tag, str) or not isinstance(count, (int, float)):
                continue
            if count < _LORA_MIN_TAG_FREQ:
                continue
            summed[tag] = summed.get(tag, 0) + int(count)
    ordered = sorted(summed.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    for tag, _count in ordered[:top_n]:
        clean = tag.translate(_LORA_TAG_NORMALIZE).strip().lower()
        if clean and clean not in out:
            out.append(clean)
    return out


def _lora_id_for_path(rel_path: str) -> str:
    """Stable ID derived from the LoRA's relative path so re-scans hit the
    same library entry. Path separators flattened to underscores so the
    safe-id regex accepts it."""
    base, _ = os.path.splitext(rel_path)
    flat = base.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return _slugify(f"lora_{flat}")[:64] or "lora_unknown"


def _scan_loras_internal(*, default_weight: float = 1.0,
                          include_triggers: bool = True,
                          use_windows_separators: bool = False,
                          refresh_existing: bool = False) -> dict:
    """Walk the configured loras directory, upsert one library entry per
    file. Returns counts + per-LoRA results. refresh_existing=True
    updates entries whose source LoRA still resolves; otherwise existing
    entries are skipped (so re-runs don't clobber user edits)."""
    if folder_paths is None:
        return {"added": 0, "updated": 0, "skipped": 0,
                "errors": ["folder_paths unavailable (running outside ComfyUI?)"]}
    try:
        names = folder_paths.get_filename_list("loras")
    except Exception as e:
        return {"added": 0, "updated": 0, "skipped": 0, "errors": [f"folder_paths failed: {e}"]}

    added = updated = skipped = 0
    errors: list[str] = []
    notify = False

    _snapshot_prompts("scan_loras")
    with _lock:
        items = _load()
        index_by_id = {i.get("id"): i for i in items}

    for rel in names:
        try:
            full_path = folder_paths.get_full_path("loras", rel)
            if not full_path or not os.path.isfile(full_path):
                continue
        except Exception as e:
            errors.append(f"{rel}: {e}")
            continue

        entry_id = _lora_id_for_path(rel)
        existed = entry_id in index_by_id
        if existed and not refresh_existing:
            skipped += 1
            continue

        # Read safetensors header once — used for both triggers and the
        # ss_base_model_version tag heuristic.
        md = _read_safetensors_metadata(full_path) if (
            include_triggers or os.path.getsize(full_path) > 0
        ) else {}

        # Compose the entry text: <lora:path:weight>, triggers
        path_token = rel.replace("/", "\\") if use_windows_separators else rel
        text = f"<lora:{path_token}:{default_weight:g}>"
        if include_triggers:
            triggers = _extract_lora_triggers(md)
            if triggers:
                text += ", " + ", ".join(triggers)

        # Display name = filename without extension, replacing path separators
        # with " / " so nested subfolder LoRAs read cleanly in the gallery.
        display_name = os.path.splitext(rel)[0].replace("/", " / ").replace("\\", " / ")

        # Tags: 'lora' + the top-level subfolder (if any) + 'sdxl' if metadata
        # hints SDXL training, since users often filter by both.
        tags: list[str] = ["lora"]
        if "/" in rel or "\\" in rel:
            top = rel.replace("\\", "/").split("/", 1)[0].lower()
            if top and top not in tags:
                tags.append(top)
        base_model = md.get("ss_base_model_version", "") if isinstance(md, dict) else ""
        if isinstance(base_model, str) and "xl" in base_model.lower():
            tags.append("sdxl")

        # Upsert: build a row dict in the storage format and write directly
        # rather than going through the multipart upsert route — we own this
        # process and need to attach an arbitrary file as the thumbnail.
        with _lock:
            items = _load()
            existing = next((i for i in items if i.get("id") == entry_id), None)
            created = existing is None
            if created:
                existing = {"id": entry_id}
                items.append(existing)
            existing["name"] = display_name
            existing["text"] = text
            existing["tags"] = tags
            existing["notes"] = (
                f"LoRA path: {rel}\n"
                f"Default weight: {default_weight:g}\n"
                f"Auto-imported by GrimmRibbity LoRA scanner."
            )
            _touch(existing, created=created)

            preview_path = _find_lora_preview(full_path)
            if preview_path:
                try:
                    with open(preview_path, "rb") as f:
                        img_bytes = f.read()
                    if len(img_bytes) <= _MAX_IMAGE_BYTES:
                        _save_thumbnail_bytes(entry_id, img_bytes)
                except OSError as e:
                    errors.append(f"{rel}: preview {preview_path}: {e}")
            _save(items)
            notify = True

        if existed:
            updated += 1
        else:
            added += 1

    if notify:
        _notify_change()
    return {"added": added, "updated": updated, "skipped": skipped, "errors": errors}


@routes.post("/prompt_library/scan_loras")
async def scan_loras_route(request):
    payload = await request.json() if request.body_exists else {}
    default_weight = float(payload.get("default_weight", 1.0) or 1.0)
    include_triggers = bool(payload.get("include_triggers", True))
    use_windows_separators = bool(payload.get("use_windows_separators", False))
    refresh_existing = bool(payload.get("refresh_existing", False))
    result = _scan_loras_internal(
        default_weight=default_weight,
        include_triggers=include_triggers,
        use_windows_separators=use_windows_separators,
        refresh_existing=refresh_existing,
    )
    return web.json_response(result)


# --------------------------------------------------------------------------
# Bulk-import the Background node's preset list into the library so users can
# browse locations through the gallery rather than picking from the node's
# combo. Re-runnable: existing entries are skipped (or refreshed) based on
# the slug derived from the preset string.
# --------------------------------------------------------------------------


def _bg_preset_short_label(preset: str, max_len: int = 60) -> str:
    """Pull a human-readable short label out of a preset string.
    'city: Tokyo street, narrow road wet with rain, neon kanji signs...'
        →  ('city', 'Tokyo street')
    The category is the prefix up to the first ':', the label is the chunk
    up to the first ',' inside the description (or the whole description
    if it has no commas, truncated to max_len)."""
    if ":" in preset:
        category, description = preset.split(":", 1)
        category = category.strip().lower()
        description = description.strip()
    else:
        category = "background"
        description = preset.strip()
    label = description.split(",", 1)[0].strip()
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip() + "…"
    return category, label


def _bg_id_for(category: str, label: str) -> str:
    return _slugify(f"bg_{category}_{label}")[:64] or "bg_unknown"


def _import_backgrounds_internal(*, refresh_existing: bool = False) -> dict:
    """Walk _BG_PRESETS and upsert one library entry per location preset.
    Skips the (none) sentinel and divider rows ('───── home / interior ─────')
    automatically. Idempotent: same slug on re-run, no clobber unless
    refresh_existing=True."""
    added = updated = skipped = 0
    errors: list[str] = []
    notify = False

    _snapshot_prompts("import_backgrounds")
    with _lock:
        items = _load()
        index_by_id = {i.get("id"): i for i in items}

    for preset in _BG_PRESETS:
        if not preset or preset == _BG_NONE:
            continue
        if preset.startswith(_BG_DIVIDER_CHAR):
            continue
        category, label = _bg_preset_short_label(preset)
        entry_id = _bg_id_for(category, label)
        existed = entry_id in index_by_id
        if existed and not refresh_existing:
            skipped += 1
            continue

        display_name = f"{category}: {label}"
        # Single 'location' tag for sorting — keeps the gallery filter row
        # tidy. The category prefix is still visible in the entry's name +
        # text, so users can search by category.
        tags = ["location"]

        with _lock:
            items = _load()
            existing = next((i for i in items if i.get("id") == entry_id), None)
            created = existing is None
            if created:
                existing = {"id": entry_id}
                items.append(existing)
            existing["name"] = display_name
            existing["text"] = preset
            existing["tags"] = tags
            existing["notes"] = (
                "Auto-imported from the GrimmRibbity Background node preset list. "
                "Update the Background node's _BG_PRESETS to add more, then re-run "
                "the import (Shift-click to refresh existing entries)."
            )
            _touch(existing, created=created)
            _save(items)
            notify = True

        if existed:
            updated += 1
        else:
            added += 1

    if notify:
        _notify_change()
    return {"added": added, "updated": updated, "skipped": skipped, "errors": errors}


@routes.get("/prompt_library/snapshots")
async def list_snapshots_route(_request):
    return web.json_response({"snapshots": _list_snapshots(), "cap": _SNAPSHOT_CAP})


@routes.post("/prompt_library/restore_snapshot")
async def restore_snapshot_route(request):
    """Restore prompts.json from a snapshot. With no name, restores the most
    recent. Returns {ok, restored_from, pre_undo_snapshot, entries}."""
    payload = await request.json() if request.body_exists else {}
    name = (payload.get("name") or "").strip()
    if not name:
        snaps = _list_snapshots()
        if not snaps:
            return web.json_response({"ok": False, "error": "no snapshots available"}, status=404)
        name = snaps[0]["name"]
    result = _restore_snapshot(name)
    if result.get("ok"):
        _notify_change()
        return web.json_response(result)
    return web.json_response(result, status=400)


@routes.post("/prompt_library/import_backgrounds")
async def import_backgrounds_route(request):
    payload = await request.json() if request.body_exists else {}
    refresh_existing = bool(payload.get("refresh_existing", False))
    result = _import_backgrounds_internal(refresh_existing=refresh_existing)
    return web.json_response(result)


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
    payload, err = await _json_payload(request)
    if err is not None:
        return err
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
    payload, err = await _json_payload(request)
    if err is not None:
        return err
    ids_raw = payload.get("ids") or []
    if not isinstance(ids_raw, list):
        return web.json_response({"error": "ids must be a list"}, status=400)
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
    payload, err = await _json_payload(request)
    if err is not None:
        return err
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
        if src.get("negative"):
            clone["negative"] = src["negative"]
        if src.get("notes"):
            clone["notes"] = src["notes"]
        if src.get("loras"):
            # Deep copy per-row so a future schema change with a nested
            # value (e.g. triggers becoming a list) doesn't make a clone
            # share state with its source. dict(l) was a shallow copy.
            clone["loras"] = [copy.deepcopy(l) for l in src["loras"]]
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
    payload, err = await _json_payload(request)
    if err is not None:
        return err
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


# --------------------------------------------------------------------------
# Library validator — gallery-side counterpart to tools/library_validate.py.
# Reuses _scan_image_ids + folder_paths so a Comfy-running install gets the
# same answers the CLI would, just via HTTP for the UI to render.
# --------------------------------------------------------------------------


def _validate_library_internal() -> dict:
    """Walk the live library + thumbnail dir + (when available) the
    folder_paths LoRA index. Returns a dict of findings the gallery's
    validator modal renders directly."""
    with _lock:
        items = _load()
    image_ids = _scan_image_ids()
    valid_entry_ids: set[str] = set()
    broken_loras: list[dict] = []
    invalid_ids: list[str] = []
    empty_texts: list[dict] = []

    available_loras: set[str] | None = None
    if folder_paths is not None:
        try:
            available_loras = set(folder_paths.get_filename_list("loras") or [])
        except Exception:
            available_loras = None

    for raw in items:
        if not isinstance(raw, dict):
            continue
        eid = raw.get("id", "")
        ename = raw.get("name", "")
        if not _safe_id(eid or ""):
            invalid_ids.append(eid)
            continue
        valid_entry_ids.add(eid)
        if not (raw.get("text") or "").strip():
            empty_texts.append({"id": eid, "name": ename})
        for l in raw.get("loras") or []:
            if not isinstance(l, dict) or not l.get("enabled", True):
                continue
            lname = (l.get("name") or "").strip()
            if not lname:
                continue
            # If folder_paths is available we can decide authoritatively;
            # otherwise we can't tell broken from intact, so we omit. The
            # client UI shows a "running outside Comfy" note in that case.
            if available_loras is not None and lname not in available_loras:
                broken_loras.append({"id": eid, "name": ename, "lora": lname})

    orphan_images = sorted(image_ids - valid_entry_ids)
    return {
        "entries_count": len(items),
        "thumbnails_count": len(image_ids),
        "broken_loras": broken_loras,
        "orphan_images": orphan_images,
        "invalid_ids": invalid_ids,
        "empty_texts": empty_texts,
        "lora_index_available": available_loras is not None,
    }


@routes.get("/prompt_library/validate")
async def validate_library(_request):
    """Gallery-side counterpart to tools/library_validate.py — surfaces
    broken LoRA references / orphan thumbnails / invalid ids / empty
    texts so the UI can render a maintenance summary. Read-only; cleanup
    actions live on /fix_orphans (and the per-entry edit/delete routes
    handle the rest)."""
    return web.json_response(_validate_library_internal())


@routes.post("/prompt_library/fix_orphans")
async def fix_orphans(request):
    """Delete every thumbnail file that doesn't correspond to a current
    library entry. Idempotent — running with no orphans returns
    {removed: 0}. Mirrors the CLI's --fix-orphans flag."""
    findings = _validate_library_internal()
    orphans = findings.get("orphan_images", [])
    removed = 0
    errors: list[str] = []
    for oid in orphans:
        if not _safe_id(oid):
            errors.append(f"unsafe orphan id skipped: {oid!r}")
            continue
        for ext in _ALLOWED_IMAGE_EXT:
            p = IMAGES_DIR / f"{oid}{ext}"
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                errors.append(f"{p.name}: {e}")
    if removed or errors:
        _notify_change()
    return web.json_response({"removed": removed, "errors": errors})


__version__ = "0.53.0"


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

try:
    from .civitai_save import CivitaiSaveImage
    _civitai_node = {"GrimmRibbityCivitaiSave": CivitaiSaveImage}
    _civitai_label = {"GrimmRibbityCivitaiSave": "GrimmRibbity — Save Image (Civitai)"}
except Exception as _e:
    print(f"[PromptLibrary] Civitai save node unavailable: {_e}")
    _civitai_node, _civitai_label = {}, {}

# SDXL sampler depends on Comfy's runtime imports (comfy.sd, comfy.samplers).
# Skipped gracefully under unittest / standalone tooling so the rest of the
# package keeps working when ComfyUI isn't on the path.
try:
    from .sampler_sdxl import (
        GrimmRibbitySamplerSDXL,
        GrimmRibbityHiResFixScript,
        GrimmRibbityPackSDXLTuple,
    )
    _sampler_node = {
        "GrimmRibbitySamplerSDXL": GrimmRibbitySamplerSDXL,
        "GrimmRibbityHiResFixScript": GrimmRibbityHiResFixScript,
        "GrimmRibbityPackSDXLTuple": GrimmRibbityPackSDXLTuple,
    }
    _sampler_label = {
        "GrimmRibbitySamplerSDXL": "GrimmRibbity — SDXL Sampler",
        "GrimmRibbityHiResFixScript": "GrimmRibbity — HiResFix Script",
        "GrimmRibbityPackSDXLTuple": "GrimmRibbity — Pack SDXL Tuple",
    }
except Exception as _e:
    print(f"[PromptLibrary] SDXL sampler unavailable: {_e}")
    _sampler_node, _sampler_label = {}, {}

try:
    from .lora_picker import GrimmRibbityLoraPicker
    _lora_node = {"GrimmRibbityLoraPicker": GrimmRibbityLoraPicker}
    _lora_label = {"GrimmRibbityLoraPicker": "GrimmRibbity — LoRA Picker"}
except Exception as _e:
    print(f"[PromptLibrary] LoRA picker unavailable: {_e}")
    _lora_node, _lora_label = {}, {}

try:
    from .sampler_anima import GrimmRibbityAnimaSampler, GrimmRibbityAnimaHiResFixScript
    _anima_node = {
        "GrimmRibbityAnimaSampler": GrimmRibbityAnimaSampler,
        "GrimmRibbityAnimaHiResFixScript": GrimmRibbityAnimaHiResFixScript,
    }
    _anima_label = {
        "GrimmRibbityAnimaSampler": "GrimmRibbity — Anima Sampler",
        "GrimmRibbityAnimaHiResFixScript": "GrimmRibbity — Anima HiResFix Script",
    }
except Exception as _e:
    print(f"[PromptLibrary] Anima sampler unavailable: {_e}")
    _anima_node, _anima_label = {}, {}

try:
    from .character_anchor import GrimmRibbityCharacterAnchor
    _anchor_node = {"GrimmRibbityCharacterAnchor": GrimmRibbityCharacterAnchor}
    _anchor_label = {"GrimmRibbityCharacterAnchor": "GrimmRibbity — Character Anchor"}
except Exception as _e:
    print(f"[PromptLibrary] Character Anchor unavailable: {_e}")
    _anchor_node, _anchor_label = {}, {}

try:
    from .detailer_node import GrimmRibbitySmartDetailer
    _detailer_node = {"GrimmRibbitySmartDetailer": GrimmRibbitySmartDetailer}
    _detailer_label = {"GrimmRibbitySmartDetailer": "GrimmRibbity — Smart Detailer"}
except Exception as _e:
    print(f"[PromptLibrary] Smart Detailer unavailable: {_e}")
    _detailer_node, _detailer_label = {}, {}

try:
    from .comic_page import GrimmRibbityComicPage
    _comic_page_node = {"GrimmRibbityComicPage": GrimmRibbityComicPage}
    _comic_page_label = {"GrimmRibbityComicPage": "GrimmRibbity — Comic Page (Regional)"}
except Exception as _e:
    print(f"[PromptLibrary] Comic Page (Regional) unavailable: {_e}")
    _comic_page_node, _comic_page_label = {}, {}

# Style node depends on torch + comfy.sd at import time, so guard the import
# the same way the sampler nodes do — keeps the rest of the package usable
# when ComfyUI isn't on the path (tests, maintenance scripts).
try:
    from .style_node import PromptLibraryStyle
    _style_node = {"PromptLibraryStyle": PromptLibraryStyle}
    _style_label = {"PromptLibraryStyle": "GrimmRibbity — Style (LoRA + Conditioning)"}
except Exception as _e:
    print(f"[PromptLibrary] Style node unavailable: {_e}")
    _style_node, _style_label = {}, {}

NODE_CLASS_MAPPINGS = {
    "PromptLibrary": PromptLibrary,
    "PromptLibraryMulti": PromptLibraryMulti,
    "PromptLibrarySave": PromptLibrarySave,
    "PromptLibraryThumbnailSaver": PromptLibraryThumbnailSaver,
    "PromptLibraryRandom": PromptLibraryRandom,
    "PromptLibraryWildcard": PromptLibraryWildcard,
    "PromptLibraryScene": PromptLibraryScene,
    "PromptLibraryBackground": PromptLibraryBackground,
    "PromptLibraryComicFrame": PromptLibraryComicFrame,
    **_civitai_node,
    **_sampler_node,
    **_lora_node,
    **_anima_node,
    **_anchor_node,
    **_detailer_node,
    **_comic_page_node,
    **_style_node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLibrary": "GrimmRibbity — Library",
    "PromptLibraryMulti": "GrimmRibbity — Multi Library (3 panels)",
    "PromptLibrarySave": "GrimmRibbity — Save",
    "PromptLibraryThumbnailSaver": "GrimmRibbity — Thumbnail Saver",
    "PromptLibraryRandom": "GrimmRibbity — Random by Tag",
    "PromptLibraryWildcard": "GrimmRibbity — Wildcard Expand",
    "PromptLibraryScene": "GrimmRibbity — Scene",
    "PromptLibraryBackground": "GrimmRibbity — Background (locked)",
    "PromptLibraryComicFrame": "GrimmRibbity — Comic Frame",
    **_civitai_label,
    **_sampler_label,
    **_lora_label,
    **_anima_label,
    **_anchor_label,
    **_detailer_label,
    **_comic_page_label,
    **_style_label,
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
