"""Smart Detailer node — named-preset storage.

Saves the current widget values of a `GrimmRibbitySmartDetailer` node as a
named preset so different scenarios (portrait hero, group/crowd, hands-only,
anime, photoreal, speed) can be one-click switched.

Storage: ``data/detailer_presets.json``, schema-versioned, atomic write.
Bundled seed presets are written on first import when the file doesn't
already exist — they live in the same file as user presets so the user
can edit or delete any of them. Once the user has any presets saved, we
never re-seed (deleting a builtin stays deleted).

Settings dict is opaque: keys are widget names, values are scalars. The
frontend resolves which widgets to apply at load time; unknown keys are
ignored (forward-compat with future widget additions). The backend stays
unaware of the widget surface so renaming a detailer parameter doesn't
require touching presets.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DETAILER_PRESETS_PATH = ROOT / "data" / "detailer_presets.json"
_DETAILER_SCHEMA_VERSION = 1

_lock = threading.Lock()


# Bundled seed presets — written on first run when no presets file exists.
# Settings dicts are deliberately sparse: only the widgets that distinguish
# the preset from the node's INPUT_TYPES defaults are present, so loading
# a preset doesn't gratuitously rewrite values the user didn't change.
_SEED_PRESETS: list[dict[str, Any]] = [
    {
        "name": "Default (reset)",
        "description": "Reset every detailer widget to its INPUT_TYPES default. "
                       "Useful as the 'undo my tweaks' button.",
        "settings": {
            "enable_face": True, "enable_eyes": True,
            "enable_hands": False, "enable_feet": False,
            "seed": 0, "steps": 25, "cfg": 6.0,
            "sampler_name": "dpmpp_3m_sde_gpu", "scheduler": "karras",
            "denoise": 0.45, "guide_size": 1024, "max_size": 1536,
            "bbox_threshold": 0.45, "max_per_target": 0,
            "tiled_decode": True, "tiled_encode": False,
            "mask_strength": 1.0, "same_seed_per_target": False, "bypass": False,
            "wildcard_prefix": "",
            "face_threshold": -1.0, "face_denoise": -1.0,
            "face_max": 0, "face_steps": 0,
            "eyes_threshold": -1.0, "eyes_denoise": -1.0,
            "eyes_max": 0, "eyes_steps": 0,
            "hands_threshold": -1.0, "hands_denoise": -1.0,
            "hands_max": 0, "hands_steps": 0,
            "feet_threshold": -1.0, "feet_denoise": -1.0,
            "feet_max": 0, "feet_steps": 0,
            "face_crop_factor": -1.0, "eyes_crop_factor": -1.0,
            "hands_crop_factor": -1.0, "feet_crop_factor": -1.0,
            "face_cycles": 1, "eyes_cycles": 1,
            "hands_cycles": 1, "feet_cycles": 1,
            "force_inpaint": True, "drop_size": 10, "min_detail_size": 48,
            "nms_iou": 0.5, "yolo_imgsz": 960, "max_bbox_area_pct": 0.95,
            "draw_preview": True,
        },
    },
    {
        "name": "Portrait Hero",
        "description": "Single subject, max-quality face + eyes pass. Caps "
                       "detections at 1 (skips background figures), longer "
                       "sample budget, slightly softer denoise to preserve "
                       "identity.",
        "settings": {
            "enable_face": True, "enable_eyes": True,
            "enable_hands": False, "enable_feet": False,
            "steps": 30, "denoise": 0.42, "cfg": 6.0,
            "max_per_target": 1, "bbox_threshold": 0.50,
            "face_crop_factor": 3.0, "eyes_crop_factor": 1.5,
            "min_detail_size": 64,
        },
    },
    {
        "name": "Group / Crowd",
        "description": "Multiple subjects in frame. Lower threshold for "
                       "inclusivity, larger drop_size to skip background "
                       "figures, all detections processed.",
        "settings": {
            "enable_face": True, "enable_eyes": True,
            "enable_hands": False, "enable_feet": False,
            "steps": 20, "denoise": 0.40,
            "max_per_target": 0, "bbox_threshold": 0.40,
            "drop_size": 30, "min_detail_size": 56,
        },
    },
    {
        "name": "Anime / Illustration",
        "description": "Stylised content — pushes eyes harder than faces, "
                       "lowers CFG to avoid over-bake, smaller YOLO inference "
                       "size since anime faces fill more of the frame.",
        "settings": {
            "enable_face": True, "enable_eyes": True,
            "enable_hands": False, "enable_feet": False,
            "steps": 25, "denoise": 0.45, "cfg": 5.5,
            "eyes_denoise": 0.50, "eyes_crop_factor": 1.7,
            "face_crop_factor": 3.5,
            "yolo_imgsz": 640,
        },
    },
    {
        "name": "Photo Realistic",
        "description": "Real-photo subjects. Gentle denoise preserves "
                       "likeness, slightly higher CFG sharpens features, "
                       "stricter threshold rejects YOLO false positives.",
        "settings": {
            "enable_face": True, "enable_eyes": True,
            "enable_hands": False, "enable_feet": False,
            "steps": 25, "denoise": 0.38, "cfg": 6.5,
            "bbox_threshold": 0.50, "mask_strength": 1.0,
            "face_crop_factor": 2.5, "eyes_crop_factor": 1.5,
        },
    },
    {
        "name": "Hands Focus",
        "description": "Face off, hands + feet on. Hands run with high "
                       "denoise (0.50) and 2 cycles since SDXL hands "
                       "frequently need geometry rebuilding rather than "
                       "refinement.",
        "settings": {
            "enable_face": False, "enable_eyes": False,
            "enable_hands": True, "enable_feet": True,
            "steps": 25, "denoise": 0.45,
            "hands_denoise": 0.50, "hands_cycles": 2,
            "hands_crop_factor": 2.0, "feet_crop_factor": 2.5,
        },
    },
    {
        "name": "Quick / Speed",
        "description": "Fastest viable detail pass. Face only, eyes off, "
                       "low steps, max=1 detection, preview drawing off, "
                       "force_inpaint off to skip already-clean crops.",
        "settings": {
            "enable_face": True, "enable_eyes": False,
            "enable_hands": False, "enable_feet": False,
            "steps": 15, "denoise": 0.40,
            "max_per_target": 1, "bbox_threshold": 0.50,
            "draw_preview": False, "force_inpaint": False,
            "min_detail_size": 64,
        },
    },
]


def _now() -> int:
    return int(time.time())


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_data_dir() -> None:
    DETAILER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _empty_store() -> dict[str, Any]:
    return {"schema_version": _DETAILER_SCHEMA_VERSION, "presets": []}


def _load_raw() -> dict[str, Any]:
    """Read the store, or return an empty store shape. Never raises."""
    if not DETAILER_PRESETS_PATH.exists():
        return _empty_store()
    try:
        with DETAILER_PRESETS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[GrimmRibbity:detailer_presets] failed to read "
              f"{DETAILER_PRESETS_PATH}: {e}; treating as empty")
        return _empty_store()
    if not isinstance(data, dict) or not isinstance(data.get("presets"), list):
        return _empty_store()
    return data


def _save_raw(data: dict[str, Any]) -> None:
    _ensure_data_dir()
    tmp = DETAILER_PRESETS_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(DETAILER_PRESETS_PATH)


def _coerce_preset(raw: Any) -> dict[str, Any] | None:
    """Normalise a stored entry; reject malformed ones (missing name/id)."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    pid = str(raw.get("id") or "").strip() or _new_id()
    settings = raw.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    return {
        "id": pid,
        "name": name[:120],
        "description": str(raw.get("description") or "")[:600],
        "builtin": bool(raw.get("builtin", False)),
        "created_at": int(raw.get("created_at") or _now()),
        "updated_at": int(raw.get("updated_at") or _now()),
        "settings": settings,
    }


def seed_default_presets() -> None:
    """Write the bundled presets when no file exists yet. Idempotent:
    once the file exists (even if empty user-deleted everything), we
    never re-seed — deletion of a builtin should stick across restarts."""
    with _lock:
        if DETAILER_PRESETS_PATH.exists():
            return
        now = _now()
        presets = []
        for seed in _SEED_PRESETS:
            presets.append({
                "id": _new_id(),
                "name": seed["name"],
                "description": seed.get("description", ""),
                "builtin": True,
                "created_at": now,
                "updated_at": now,
                "settings": dict(seed.get("settings", {})),
            })
        _save_raw({
            "schema_version": _DETAILER_SCHEMA_VERSION,
            "presets": presets,
        })


def list_presets() -> list[dict[str, Any]]:
    """Return preset metadata (no settings) for the picker."""
    with _lock:
        data = _load_raw()
    out = []
    for raw in data.get("presets", []):
        p = _coerce_preset(raw)
        if p is None:
            continue
        out.append({
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "builtin": p["builtin"],
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
            "setting_count": len(p["settings"]),
        })
    out.sort(key=lambda p: ((0 if p["builtin"] else 1), p["name"].lower()))
    return out


def get_preset(preset_id: str) -> dict[str, Any] | None:
    """Return a full preset including settings, or None."""
    if not preset_id:
        return None
    with _lock:
        data = _load_raw()
    for raw in data.get("presets", []):
        p = _coerce_preset(raw)
        if p is None:
            continue
        if p["id"] == preset_id:
            return p
    return None


def upsert_preset(
    name: str,
    settings: dict[str, Any],
    description: str = "",
    preset_id: str | None = None,
) -> dict[str, Any]:
    """Save a preset. When preset_id is provided, updates that entry
    in place (preserving created_at). Otherwise creates a new one;
    if a preset with the same name (case-insensitive) already exists,
    that entry is updated instead of duplicating. Returns the saved
    preset."""
    name = (name or "").strip()
    if not name:
        raise ValueError("preset name required")
    if not isinstance(settings, dict):
        raise ValueError("settings must be a dict")
    with _lock:
        data = _load_raw()
        presets = data.get("presets", [])
        now = _now()
        target_idx = -1
        if preset_id:
            for i, raw in enumerate(presets):
                if isinstance(raw, dict) and raw.get("id") == preset_id:
                    target_idx = i
                    break
        if target_idx < 0:
            lower = name.lower()
            for i, raw in enumerate(presets):
                if isinstance(raw, dict) and str(raw.get("name") or "").strip().lower() == lower:
                    target_idx = i
                    break
        if target_idx >= 0:
            existing = presets[target_idx] or {}
            saved = {
                "id": existing.get("id") or _new_id(),
                "name": name[:120],
                "description": description[:600],
                "builtin": False,
                "created_at": int(existing.get("created_at") or now),
                "updated_at": now,
                "settings": dict(settings),
            }
            presets[target_idx] = saved
        else:
            saved = {
                "id": _new_id(),
                "name": name[:120],
                "description": description[:600],
                "builtin": False,
                "created_at": now,
                "updated_at": now,
                "settings": dict(settings),
            }
            presets.append(saved)
        data["presets"] = presets
        data["schema_version"] = _DETAILER_SCHEMA_VERSION
        _save_raw(data)
        return saved


def delete_preset(preset_id: str) -> bool:
    """Remove a preset. Builtins are not protected — user can delete
    anything; seed_default_presets won't restore them unless the
    entire file is gone."""
    if not preset_id:
        return False
    with _lock:
        data = _load_raw()
        presets = data.get("presets", [])
        n = len(presets)
        data["presets"] = [
            raw for raw in presets
            if not (isinstance(raw, dict) and raw.get("id") == preset_id)
        ]
        if len(data["presets"]) == n:
            return False
        _save_raw(data)
        return True
